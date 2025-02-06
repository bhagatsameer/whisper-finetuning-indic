# original: https://huggingface.co/blog/fine-tune-whisper

"""Finetune Whisper on custom dataset"""

import os
import io
import argparse
import random

import torch
import numpy as np
import evaluate
import wandb
import webdataset as wds

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets, load_from_disk
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig,
    GenerationConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from indicnlp import common
from indicnlp import loader
from indicnlp.normalize.indic_normalize import DevanagariNormalizer
from huggingface_hub import get_token

from torch.utils.data import DataLoader
from tqdm import tqdm

hf_token = get_token()

AUDIO_COLUMN_NAME = "audio"
TEXT_COLUMN_NAME = "sentence"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that pads input features and labels for both WebDataset and HuggingFace datasets.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Union[tuple, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collates features for both WebDataset and HuggingFace datasets:
        - Ensures `input_features` and `labels` are uniformly processed and padded.
        """
        if isinstance(features[0], tuple):  # WebDataset case
            input_features, label_features = zip(*features)
            input_features = [f for f in input_features]
            labels = [l for l in label_features]

        else:  # HuggingFace dataset case
            input_features = [feature["input_features"] for feature in features]
            labels = [feature["labels"] for feature in features]

        input_features = [{"input_features": f} for f in input_features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": l} for l in labels]

        # Pad and truncate labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Cut BOS token if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def preprocess_webdataset(sample: Dict[str, Any]) -> Union[Dict[str, Any], None]:
    """
    Preprocess a single sample from a WebDataset.
    Args:
        sample: A dictionary containing raw data from WebDataset.
    Returns:
        A dictionary with 'input_features' and 'labels' ready for training, or None if an error occurs.
    """
    try:
        input_features = np.load(io.BytesIO(sample["input.npz"]))["input_features"].tolist()  # Convert to list
        labels = np.load(io.BytesIO(sample["labels.npz"]))["labels"].tolist()  # Convert to list

        return {
            "input_features": input_features,
            "labels": labels,
        }
    except Exception as e:
        sample_key = sample.get("__key__", "unknown")
        print(f"Error processing sample {sample_key}: {e}")
        return None  # Skip corrupted sample


def load_webdataset(shard_pattern: str):
    """
    Load a WebDataset using a shard pattern and preprocess each sample.
    Args:
        shard_pattern: File pattern for the shards.
    Returns:
        A WebDataset iterator yielding tuples (input_features, labels).
    """
    dataset = (
        wds.WebDataset(shard_pattern, shardshuffle=False)
        .map(preprocess_webdataset)             # Apply preprocessing
        .to_tuple("input_features", "labels")   # Select keys
    )

    return dataset


def train(opt):
    """
    Main training function that sets up the model, data loaders, training arguments, and initiates training.
    
    Args:
        opt: Parsed command-line options.
    """
    language = opt.language
    device = opt.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_size = opt.model_size
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model_size}")

    tokenizer = WhisperTokenizer.from_pretrained(
        f"openai/whisper-{model_size}",
        language=language, 
        task="transcribe"
    )
    processor = WhisperProcessor(feature_extractor, tokenizer)

    metric = evaluate.load("wer")    
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.generation_config.language = language
    model.generation_config.task = "transcribe"

    model.generation_config.forced_decoder_ids = None
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(pred):
        """
        Compute Word Error Rate (WER) metrics for predictions.
        
        Args:
            pred: Prediction output containing predictions and label_ids.
        Returns:
            A dictionary with the WER value.
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        if normalizer:
            pred_str = [normalizer.normalize(pred) for pred in pred_str]
            label_str = [normalizer.normalize(label) for label in label_str]
        else:
            pred_str = [processor.tokenizer._normalize(pred) for pred in pred_str]
            label_str = [processor.tokenizer._normalize(label) for label in label_str]
        
        # filtering step to only evaluate the samples that correspond to non-zero references
        pred_str = [pred_str[i].strip() for i in range(len(pred_str)) if len(label_str[i]) > 0]
        label_str = [label_str[i].strip() for i in range(len(label_str)) if len(label_str[i]) > 0]

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    if opt.use_webdataset:
        if os.path.exists(opt.webdataset_path_or_url):
            train_shard_pattern = f"{opt.webdataset_path_or_url}/shard-{{{opt.train_shards}}}.tar"
            eval_shard_pattern = f"{opt.webdataset_path_or_url}/shard-{{{opt.eval_shards}}}.tar"
        else:
            url = f"{opt.webdataset_path_or_url}/shard-{{{opt.train_shards}}}.tar"
            train_shard_pattern = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'"

            url = f"{opt.webdataset_path_or_url}/shard-{{{opt.eval_shards}}}.tar"
            eval_shard_pattern = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'"

        train_dataset = load_webdataset(train_shard_pattern)
        eval_dataset = load_webdataset(eval_shard_pattern)
    else:
        train_dataset = load_from_disk(opt.hf_train_dataset_path)
        eval_dataset = load_from_disk(opt.hf_test_dataset_path)

    if opt.use_wandb:
        wandb.init(project=opt.wandb_project_name, name=opt.wandb_run_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./whisper-{model_size}-{language}-finetuned-{opt.wandb_run_name}",  # change to a repo name of your choice
        per_device_train_batch_size=opt.batch_size,
        gradient_accumulation_steps=opt.grad_acc,  # increase by 2x for every 2x decrease in batch size
        learning_rate=opt.learning_rate,
        weight_decay=0.05,
        warmup_steps=opt.warmup_steps,
        max_steps=opt.max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        dataloader_pin_memory=True,
        save_steps=opt.save_steps,
        eval_steps=opt.eval_steps,
        logging_steps=25,
        save_total_limit=5,
        report_to=["tensorboard", "wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        ignore_data_skip=True,
        optim="adamw_bnb_8bit",
        save_safetensors=False,
        dataloader_num_workers=opt.num_workers)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=opt.resume_from_checkpoint)
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', default="small", type=str, help='whisper model size.')
    parser.add_argument('--language', default="hi", type=str, help='Dataset language.')
    parser.add_argument('--batch-size', default=16, type=int, help='Batch-size per device.')
    parser.add_argument('--grad-acc', default=4, type=int, help='Gradient accumulation steps.')
    parser.add_argument('--learning_rate', '-lr', default=3e-5, type=float, help='Learning rate.')
    parser.add_argument('--max_steps', default=10000, type=int, help='Max training steps.')
    parser.add_argument('--warmup_steps', default=500, type=int, help='Warmup training steps.')
    parser.add_argument('--save_steps', default=1000, type=int, help='Number steps to save model checkpoint.')
    parser.add_argument('--eval_steps', default=1000, type=int, help='Number of steps to do evaluation.')
    parser.add_argument('--device', default=None, type=str, help='Device to run training.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of dataloader workers')
    parser.add_argument('--resume_from_checkpoint', default=False, action='store_true',
                        help="Resume training from the latest checkpoint if available.")
    
    parser.add_argument('--use_webdataset', default=False, action='store_true', help="Use Webdataset.")
    parser.add_argument('--webdataset_path_or_url', '-wds', default=None, type=str,
                        help="Path or URL for the preprocessed WebDataset. Can be a local path or remote URL.")
    parser.add_argument('--train_shards', type=str, help="Shard range for training, e.g., '00000..00269'.")
    parser.add_argument('--eval_shards', type=str, help="Shard range for evaluation, e.g., '00270..00271'.")
    parser.add_argument('--hf_train_dataset_path', default=None, type=str,
                        help="Path for the preprocessed Huggingface train Dataset. Should be a local path or hf dataset url.")
    parser.add_argument('--hf_test_dataset_path', default=None, type=str,
                        help="Path for the preprocessed Huggingface test Dataset. Should be a local path or hf dataset url.")

    parser.add_argument('--use_wandb', default=False, action='store_true', help="Log training metrics on W&B.")
    parser.add_argument('--wandb_project_name', default="whisper-hindi", type=str, help='W&B project name.')
    parser.add_argument('--wandb_run_name', default="run0-small", type=str, help='W&B run name.')
    parser.add_argument('--use_indic_norm', default=False, action='store_true', help="Use Indic normalization for text processing.")

    opt = parser.parse_args()

    if opt.use_webdataset:
        if not opt.webdataset_path_or_url:
            raise ValueError("You must specify --webdataset_path_or_url to provide the location of preprocessed WebDataset.")
        if not opt.train_shards or not opt.eval_shards:
            raise ValueError("You must specify both --train-shards and --eval-shards.")
    elif opt.hf_train_dataset_path is None or opt.hf_test_dataset_path is None:
        raise ValueError(f"Either `--use_webdataset` should be set to True or provide a valid preprocessed hf train and test datasets.")

    if opt.use_wandb and (not opt.wandb_project_name or not opt.wandb_run_name):
        raise ValueError("You must specify --wandb_project_name and --wandb_run_name when using W&B logging.")

    print(f"Starting training with the following parameters:")
    for arg, value in vars(opt).items():
        print(f"{arg}: {value}")

    if opt.use_indic_norm:
        common.set_resources_path('indic_nlp_resources')
        loader.load()

        normalizer = DevanagariNormalizer(opt.language)
    else:
        normalizer = None
    
    train(opt)

if __name__=="__main__":
    main()
