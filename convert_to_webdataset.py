import torch
import os
import numpy as np
import argparse
import tarfile
import io

import webdataset as wds
from tqdm import tqdm
from multiprocessing import Pool
from datasets import load_from_disk, concatenate_datasets


def save_shard(output_dir, shard_idx, shard_samples):
    """
    Save a single shard containing multiple samples to a tar archive.
    Args:
        output_dir: Directory where shards are saved.
        shard_idx: Index of the shard.
        shard_samples: List of samples to include in the shard.
    """
    shard_path = os.path.join(output_dir, f"shard-{shard_idx:05d}.tar")
    with wds.TarWriter(shard_path) as sink:
        for sample_idx, sample in enumerate(shard_samples):
            sample_key = f"sample{sample_idx:06d}"
            if np.isnan(sample["input_features"]).any() or np.isinf(sample["input_features"]).any():
                print(f"Skipping sample {sample_key} due to NaN or Inf in input_features.")
                continue

            input_buffer = io.BytesIO()
            np.savez_compressed(input_buffer, input_features=sample["input_features"])
            input_buffer.seek(0)

            label_buffer = io.BytesIO()
            np.savez_compressed(label_buffer, labels=sample["labels"])
            label_buffer.seek(0)

            sink.write({
                "__key__": sample_key,                  # Unique identifier for the sample
                "input.npz": input_buffer.getvalue(),   # Input features saved as .npz bytes
                "labels.npz": label_buffer.getvalue(),  # Labels saved as .npz bytes
            })


def process_shard_range(args):
    """
    Process a range of the dataset and save shards.
    Args:
        args: Tuple containing (dataset_path, shard_indices, output_dir, shard_size).
    """
    dataset, shard_indices, output_dir, shard_size = args
    for shard_idx, start_idx in shard_indices:
        end_idx = start_idx + shard_size
        shard_samples = dataset.select(range(start_idx, min(len(dataset), end_idx)))
        save_shard(output_dir, shard_idx, shard_samples)


def create_webdataset(dataset, output_dir, shard_size, num_proc, shard_start_idx):
    """
    Process dataset into WebDataset shards using multiprocessing.
    Args:
        dataset_path: Path to the dataset.
        output_dir: Directory to save tar shards.
        shard_size: Number of samples per shard.
        num_proc: Number of processes to use.
        shard_start_idx: Starting index for shard naming.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset to get the total number of samples
    total_samples = len(dataset)
    print(f"Total samples in dataset: {total_samples}")
    total_shards = (total_samples + shard_size - 1) // shard_size

    shard_indices = [
        (shard_start_idx + i, i * shard_size)
        for i in range(total_shards)
    ]

    num_proc = min(num_proc, len(shard_indices))
    shard_ranges = np.array_split(shard_indices, num_proc)

    args = [
        (dataset, shard_range, output_dir, shard_size)
        for shard_range in shard_ranges
    ]

    with Pool(num_proc) as pool:
        list(tqdm(pool.imap(process_shard_range, args), total=num_proc, desc="Processing shards"))

# Main script with chunked processing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a dataset into WebDataset tar shards in memory-efficient chunks.")
    parser.add_argument("--preprocessed_datasets", type=str, required=True, help="Path to the dir with preprocessed_datasets")
    parser.add_argument("--output_dir", type=str, default="./webdataset-hi", help="Directory to save tar shards.")
    parser.add_argument("--shard_size", type=int, default=1000, help="Number of samples per shard.")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for multiprocessing.")
    parser.add_argument("--shard_start_idx", type=int, default=0, help="Starting index for shard naming.")
    args = parser.parse_args()

    # Ideally, we want to create the webdataset with samples already shuffled
    try:
        # Load all datasets from the directory and concatenate them
        print(f"Loading datasets from: {args.preprocessed_datasets}")
        datasets = [
            load_from_disk(os.path.join(args.preprocessed_datasets, dataset_dir))
            for dataset_dir in os.listdir(args.preprocessed_datasets)
            if os.path.isdir(os.path.join(args.preprocessed_datasets, dataset_dir))
        ]
        concatenated_dataset = concatenate_datasets(datasets)
        print("Shuffling dataset...")
        concatenated_dataset = concatenated_dataset.shuffle(seed=42)

    except Exception as e:
        print(f"Error loading datasets: {e}")
        import sys
        sys.exit(1)

    # Create WebDataset shards
    create_webdataset(
        concatenated_dataset,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        num_proc=args.num_proc,
        shard_start_idx=args.shard_start_idx,
    )
