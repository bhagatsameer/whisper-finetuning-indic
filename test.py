import argparse
import os
import re

import torch
import evaluate
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

from indicnlp import common
from indicnlp import loader
from indicnlp.normalize.indic_normalize import DevanagariNormalizer


AUDIO_COLUMN_NAME = "audio"
TEXT_COLUMN_NAME = "sentence"


def is_length_hallucination(pred_text, ref_text, ratio_threshold=1.5):
    pred_len = len(pred_text.split())
    ref_len = len(ref_text.split())
    return pred_len > ratio_threshold * ref_len

def validate(model_path, dataset, opt, language="hi", whisper_norm=True, model_size="tiny"):
    """
    Validate the Whisper model on the provided test dataset and compute WER.
    
    Args:
        model_path (str): Path to the finetuned model or model identifier.
        dataset (DatasetDict): Dataset dictionary containing the "test" split.
        opt (Namespace): Parsed command-line options.
        language (str): Language code.
        whisper_norm (bool): Whether to use Whisper's inbuilt normalization.
        model_size (str): Size of the Whisper model.

    Returns:
        float: The average Word Error Rate (WER) on the test dataset.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Initialize feature extractor, tokenizer and processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model_size}")
    tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-{model_size}", language=language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}", language=language, task="transcribe")

    metric = evaluate.load("wer")

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    model.config.suppress_tokens = []

    def compute_metrics(pred, do_normalize_eval=True, wnorm=True):
        """
        Compute the WER for a batch of predictions.
        
        Args:
            pred (dict): Dictionary with "predictions" and "label_ids".
            do_normalize_eval (bool): Whether to perform normalization.
            wnorm (bool): If True, use Whisper normalization; else use Indic normalization.
        
        Returns:
            dict: A dictionary with the computed WER.
        """
        pred_ids = pred["predictions"]
        label_ids = pred["label_ids"]

        # Replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        if do_normalize_eval:
            if wnorm:
                pred_str = [processor.tokenizer._normalize(pred) for pred in pred_str]
                label_str = [processor.tokenizer._normalize(label) for label in label_str]
            else:
                pred_str = [normalizer.normalize(pred) for pred in pred_str]
                label_str = [normalizer.normalize(label) for label in label_str]

        # Filter out empty references and strip whitespaces
        pred_str = [pred_str[i].strip() for i in range(len(pred_str)) if len(label_str[i]) > 0]
        label_str = [label_str[i].strip() for i in range(len(label_str)) if len(label_str[i]) > 0]

        if is_length_hallucination(pred_str[0], label_str[0]):
            return False

        wer_value = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer_value}

    total_wer = 0.0
    num_batches = 0
    if opt.remove_digits:
        dataset["test"] = dataset["test"].filter(lambda x: not re.search(r'\d', x["sentence"]))

    for batch in dataset["test"]:
        inputs = processor(batch["audio"]["array"], return_tensors="pt")
        input_features = inputs.input_features

        generated_ids = model.generate(
            inputs=input_features.to(device),
            forced_decoder_ids=forced_decoder_ids,
            repetition_penalty=1.15,
            num_beams=5
        ).cpu()

        label_ids = tokenizer(batch["sentence"]).input_ids
        label_features = [{"input_ids": label_ids}]
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding tokens with -100 so they are ignored in loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove the bos token if present (it's appended later)
        if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        preds = {"predictions": generated_ids, "label_ids": labels}
        wer_ = compute_metrics(preds, wnorm=whisper_norm)
        if wer_:
            total_wer += wer_["wer"]
            num_batches += 1

    return total_wer / num_batches if num_batches > 0 else 0.0


def normalize_dataset(ds, audio_column_name=None, text_column_name=None):
    """
    Normalize the dataset by renaming columns, resampling audio, and removing extra columns.
    
    Args:
        ds (Dataset): The dataset to normalize.
        audio_column_name (str, optional): The current name of the audio column.
        text_column_name (str, optional): The current name of the text column.
    
    Returns:
        Dataset: The normalized dataset.
    """
    if audio_column_name is not None and audio_column_name != AUDIO_COLUMN_NAME:
        ds = ds.rename_column(audio_column_name, AUDIO_COLUMN_NAME)
    if text_column_name is not None and text_column_name != TEXT_COLUMN_NAME:
        ds = ds.rename_column(text_column_name, TEXT_COLUMN_NAME)
    
    # Resample audio to a consistent sampling rate
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    
    # Remove all columns except for "audio" and "sentence"
    ds = ds.remove_columns(set(ds.features.keys()) - {AUDIO_COLUMN_NAME, TEXT_COLUMN_NAME})
    return ds


def load_datasets(dataset_name, dataset_split, dataset_language):
    """
    Load and normalize the dataset for evaluation.
    
    Currently, only the 'google/fleurs' dataset is supported.
    
    Args:
        dataset_name (str): The name of the dataset.
        dataset_split (str): The split(s) to load.
        dataset_language (str): The language code for the dataset.
    
    Returns:
        DatasetDict: A dictionary containing the test split.
    """
    ds = DatasetDict()
    if dataset_name == "google/fleurs":
        ds_test = load_dataset(dataset_name, dataset_language, split=dataset_split)
        ds_test = normalize_dataset(ds_test, text_column_name="transcription")
    else:
        raise ValueError(
            f"{dataset_name} is not supported by the script, please add the code to load the dataset correctly. "
            "Supported datasets: [`google/fleurs`]"
        )
    ds["test"] = ds_test
    return ds


def main():
    """
    Main function to parse arguments, load datasets, and validate the Whisper model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', default="tiny", type=str, help='Whisper model size')
    parser.add_argument('--model_path', default=None, type=str, help="Finetuned model path. If None, uses openai/whisper-{model_size}")
    parser.add_argument('--language', default='hi', help="Model language code")
    parser.add_argument('--use_indic_norm', default=False, action='store_true', help="Use Indic normalization for text processing.")
    parser.add_argument('--eval_dataset', default='google/fleurs', help="Evaluation dataset.")
    parser.add_argument('--eval_split', default='train+test+validation', help="Dataset split to use for evaluation. Defaults to full dataset.")
    parser.add_argument('--dataset_language', default='hi_in', help="Evaluation dataset language code (if applicable).")
    parser.add_argument('--remove_digits', default=False, action='store_true', help="Remove samples with digits.")

    opt = parser.parse_args()

    # Setup Indic normalization if enabled
    if opt.use_indic_norm:
        if not os.path.exists("./indic_nlp_resources"):
            raise ValueError("Please clone `indic_nlp_resources` to use Indic normalizer.")
        if opt.language in ["hi"]:
            common.set_resources_path('indic_nlp_resources')
            loader.load()
            global normalizer  # Make normalizer global for use in compute_metrics
            normalizer = DevanagariNormalizer(opt.language)
        else:
            raise ValueError(f"Invalid language {opt.language} for Indic normalizer")
    
    # If mode_path, infer from model_size and use default whisper
    if opt.model_path is None:
        print(f"Using openai/whisper-{opt.model_size}")
        opt.model_path = f"openai/whisper-{opt.model_size}"
    else:
        print(f"Using {opt.model_path}")

    ds = load_datasets(opt.eval_dataset, opt.eval_split, dataset_language=opt.dataset_language)
    wer = validate(
        opt.model_path,
        ds,
        opt,
        language=opt.language,
        whisper_norm=not opt.use_indic_norm,
        model_size=opt.model_size
    )
    
    print(f"WER on {opt.eval_dataset} with language {opt.language}: {wer}")


if __name__ == "__main__":
    main()
