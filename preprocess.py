import os
import argparse
import shutil

from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
)
from indicnlp.normalize.indic_normalize import DevanagariNormalizer
from indicnlp import loader, common

AUDIO_COLUMN_NAME = "audio"
TEXT_COLUMN_NAME = "sentence"


def preprocess_dataset(dataset, processor, feature_extractor, save_dir):
    """
    Preprocess a dataset for Whisper fine-tuning.
    
    This function applies feature extraction on the audio data and converts the text transcription 
    into token IDs. It also filters out samples with empty transcriptions and samples whose labels 
    are too long.
    
    Args:
        dataset (DatasetDict): The dataset to preprocess.
        processor (WhisperProcessor): Processor containing the tokenizer and feature extractor.
        feature_extractor (WhisperFeatureExtractor): Feature extractor for audio inputs.
        save_dir (str): Directory to save the preprocessed dataset.
    """
    def prepare_dataset(batch):
        audio = batch["audio"]
        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        # encode target text to label ids
        if normalizer is not None:
            input_str = normalizer.normalize(batch[TEXT_COLUMN_NAME]).strip()
        else:
            input_str = processor.tokenizer._normalize(batch[TEXT_COLUMN_NAME]).strip()

        batch["labels"] = processor.tokenizer(input_str).input_ids
        return batch
    
    def is_labels_none(labels):
        return labels is not None and len(labels) > 0
    
    def is_labels_in_length_range(labels):
        return len(labels) < 448


    print("Applying preprocessing...")

    dataset = dataset.filter(
        is_labels_none, num_proc=8, input_columns=[TEXT_COLUMN_NAME]
    )
    dataset = dataset.map(
            prepare_dataset, remove_columns=dataset.column_names)
    dataset = dataset.filter(
        is_labels_in_length_range, num_proc=8, input_columns=["labels"]
    )
    
    # Save the preprocessed dataset to disk.
    dataset.save_to_disk(save_dir)
    print(f"Preprocessed dataset saved to: {save_dir}")


def normalize_dataset(ds, audio_column_name=None, text_column_name=None):
    """
    Normalize the dataset columns and audio sampling rate.
    
    Renames the specified audio and text columns to standard names, casts the audio column to a 
    consistent sampling rate (16kHz), and removes all columns except the audio and text columns.
    
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

    # Cast audio column to a fixed sampling rate.
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    
    # Keep only the audio and sentence columns.
    ds = ds.remove_columns(set(ds.features.keys()) - {AUDIO_COLUMN_NAME, TEXT_COLUMN_NAME})
    return ds


def load_dataset_from_source(
        dataset_type,
        dataset_name,
        split,
        data_dir=None,
        text_column_name="sentence",
        audio_column_name="audio",
        language="hi"
    ):
    """
    Load a dataset from either HuggingFace or a local audio folder.
    
    This function supports loading datasets using either the HuggingFace datasets library or from a 
    local directory with an audiofolder format.
    
    Args:
        dataset_type (str): Type of dataset to load: 'huggingface' or 'audiofolder'.
        dataset_name (str): Name of the HuggingFace dataset (used if dataset_type is 'huggingface').
        split (str): Dataset split to load (e.g., "train", "validation", "test").
        data_dir (str): Directory containing the local audiofolder dataset (used if dataset_type is 'audiofolder').
        text_column_name (str): Name of the column containing text transcriptions.
        audio_column_name (str): Name of the column containing audio data.
        language (str): Language code to pass (if applicable).
    
    Returns:
        DatasetDict: The loaded and normalized dataset.
    """
    if dataset_type == "huggingface":
        print(f"Loading HuggingFace dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, language, split=split, token=True)
    elif dataset_type == "audiofolder":
        if not data_dir:
            raise ValueError("For 'audiofolder', you must specify --data_dir.")
        print(f"Loading local audiofolder dataset from: {data_dir}")
        dataset = load_dataset("audiofolder", data_dir=data_dir, split=split)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    dataset = normalize_dataset(dataset, audio_column_name, text_column_name)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True,
                        choices=["huggingface", "audiofolder"],
                        help="Type of dataset to preprocess: 'huggingface' or 'audiofolder'.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name of the HuggingFace dataset (for 'huggingface').")
    parser.add_argument("--text_column_name", type=str, default=None,
                        help="Name of the column containing transcriptions in the dataset.")
    parser.add_argument("--audio_column_name", type=str, default=None,
                        help="Name of the column containing audio in the dataset.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the local audiofolder dataset (for 'audiofolder').")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to load (default: 'train').")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save the preprocessed dataset.")
    parser.add_argument("--model_size", default="small", type=str,
                        help="Whisper model size for processor.")
    parser.add_argument("--language", default="hi", type=str,
                        help="Language to use when processing the dataset.")
    parser.add_argument('--use_indic_norm', default=False, action='store_true',
                        help="Use Indic normalization for text processing.")
    opt = parser.parse_args()

    # Initialize tokenizer, feature extractor, and processor for Whisper.
    tokenizer = WhisperTokenizer.from_pretrained(
        f"openai/whisper-{opt.model_size}",
        language=opt.language,
        task="transcribe"
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        f"openai/whisper-{opt.model_size}"
    )
    processor = WhisperProcessor(feature_extractor, tokenizer)

    # Remove the save directory if it already exists.
    if os.path.exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)

    # Setup Indic normalization if enabled.
    global normalizer
    if opt.use_indic_norm and opt.language in ["hi"]:
        if not os.path.exists("./indic_nlp_resources"):
            raise ValueError("Please clone `indic_nlp_resources` to use Indic normalizer.")
        common.set_resources_path('indic_nlp_resources')
        loader.load()
        normalizer = DevanagariNormalizer(opt.language)
    else:
        normalizer = None

    # Load and normalize the dataset.
    dataset = load_dataset_from_source(
        dataset_type=opt.dataset_type,
        dataset_name=opt.dataset_name,
        split=opt.split,
        data_dir=opt.data_dir,
        text_column_name=opt.text_column_name,
        audio_column_name=opt.audio_column_name,
        language=opt.language
    )

    # Preprocess and save the dataset.
    preprocess_dataset(
        dataset=dataset,
        processor=processor,
        feature_extractor=feature_extractor,
        save_dir=opt.save_dir
    )


if __name__ == "__main__":
    main()