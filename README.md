# Whisper Finetuning

### Overview

This repo provides the scripts to fine-tune `openai/whisper`. The repository includes tools for data preprocessing, converting data to WebDataset format, and fine-tuning whisper.


### Datasets
The model was fine-tuned on the following datasets:

- **Shrutilipi (AI4Bharat)**
A corpus of 6400+ hours across 12 Indian languages.
Hindi subset contains approximately 1600 hours.

- **IITM Madras SpringLab**
Contains mock conversations and monologues on diverse topics.
Hindi subset covers approximately 900 hours.

- **Mozilla Foundation’s Common Voice 11.0**
Released under CC-1.0, adding robustness through community-contributed voice data.

- **Google Fleurs**
Used for testing, providing a comprehensive benchmark with its extensive multilingual dataset, licensed under CC-4.0.


### Getting Started
1. **System Requirements**
- **CUDA**: >= 11.8
- **Python**: >= 3.9
- **Hardware**: NVIDIA 4090 recommended for training or any other NVIDIA GPU for acceleration.
- **Docker Environment**: We used the Docker container `pytorch/pytorch:2.5.0-cuda12.1-cudnn9-devel` for our environment.

2. **Install Requirements**

```bash
$ pip install -r requirements.txt
```

3. **Normalization**

For using Indic normalization, clone the `indic_nlp_resources` repository inside `whisper-finetuning`:

```bash
$ git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
```
In Hindi, the meaning of a sentence can drastically change if diacritics or characters are removed, making normalization a critical
part of the pipeline. Consider this example:

```
हमने उस उम्मीदवार को चुना।
```

- Whisper's Default Normalization:

Whisper applies aggressive text normalization, often stripping diacritics and compressing words. Here's how the same sentence looks
after Whisper normalization:
```
'हमन उस उमम दव र क चन'
```
The removal of diacritics and loss of word boundaries result in text that is difficult to interpret and often meaningless.

- Indic Normalization to the Rescue:

Instead of Whisper's default normalization, we employed
[Indic Normalization from the IndicNLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library/blob/4cead0ae6c78fe9a19a51ef679f586206df9c476/indicnlp/normalize/indic_normalize.py#L325), which retains diacritics and complex characters, producing more linguistically accurate transcriptions:

```
हमने उस उम्मीदवार को चुना।
```

While Whisper's default normalization might reduce Word Error Rate (WER) on numeric benchmarks, it sacrifices semantic accuracy. For Hindi,
maintaining diacritics and preserving complex characters is vital for transcription quality, even if it slightly increases the WER. This
trade-off ensures that the transcriptions are meaningful and contextually accurate.


### Data Preprocessing

Preprocessing requires the dataset to be in Hugging Face audio dataset format. Please refer to the Hugging Face documentation for details on preparing your data.

Since Whisper's maximum audio sample duration is 30 seconds, ensure that all audio samples are around 30 seconds:

- If they are below 30 seconds, merge samples to reach ~30 seconds.
- If they are over 30 seconds, either discard them or cut them short.

And also make similar adjustments to their corresponding transcripts. Once we have the dataset in huggingface format we go ahead and preprocess the datasets i.e., preprocess all your huggingface format datasets(if you have multiple sources of data) and put them into a directory for e.g. `preprocessed_hf_datasets`:

```bash
$ python preprocess.py \
        --dataset_type huggingface \
        --dataset_name mozilla-foundation/common_voice_11_0 \
        --language hi \
        --split train \
        --save_dir ./preprocessed_hf_datasets/train/mozilla_common_voice_11_0 \
        --use_indic_norm
```
This command preprocesses the Common Voice 11.0 Hindi dataset by ensuring the audio samples are the right duration, applying Indic normalization (if enabled), and saving the processed data to the specified directory.


### Webdataset Conversion [Optional]

After preprocessing all your train & dev datasets, convert them into a shuffled webdataset format which is crucial for faster training when using large datasets.

This step is optional and can be used if you have a large dataset i.e., thousands of hours of data otherwise stick to Hugging Face format, use the preprocessed dataset saved to the disk in the previous
step.

- Write train shards

```bash
$ python convert_to_webdataset.py \
        --preprocessed_datasets ./preprocessed_hf_datasets/train/ \
        --output_dir ./webdataset_hi \
        --shard_size 1000 \
        --num_proc 4 \
        --shard_start_idx 0
```
Lets assume that the train dataset had 50000 samples and we now have written 50 shards.

- Write evaluation shards and set `--shard_start_idx=51`

```bash
$ python convert_to_webdataset.py \
        --preprocessed_datasets ./preprocessed_hf_datasets/test/ \
        --output_dir ./webdataset_hi \
        --shard_size 1000 \
        --num_proc 4 \
        --shard_start_idx 51
```
Lets assume we have 2 test shards


### Start training !!!

This command initiates the training process using the fine-tuning parameters provided, including the use of WebDataset data, specified model size, and Indic normalization. It also integrates with Weights & Biases for experiment tracking.

- Train with webdataset:

```bash
$ python train.py \
        --model_size tiny \
        --language hi \
        --batch_size 32 \
        --grad_acc 4 \
        --learning_rate 3e-5 \
        --warmup_steps 500 \
        --max_steps 10000 \
        --save_steps 1000 \
        --eval_steps 1000 \
        --use_webdataset \
        --webdataset_path_or_url ./webdataset_hi \
        --train_shards 00000..00050 \
        --eval_shards 00051..00052 \
        --use_wandb \
        --wandb_project_name whisper-hindi \
        --wandb_run_name run0-tiny \
        --use_indic_norm
```

- Train with preprocessed huggingface train & test datasets:

```bash
$ python train.py \
        --model_size tiny \
        --language hi \
        --batch_size 32 \
        --grad_acc 4 \
        --learning_rate 3e-5 \
        --warmup_steps 500 \
        --max_steps 10000 \
        --save_steps 1000 \
        --eval_steps 1000 \
        --hf_train_dataset_path ./preprocessed_hf_datasets/train/mozilla_common_voice_11_0 \
        --hf_test_dataset_path ./preprocessed_hf_datasets/test/mozilla_common_voice_11_0 \
        --use_wandb \
        --wandb_project_name whisper-hindi \
        --wandb_run_name run0-tiny \
        --use_indic_norm
```

### Evaluation

Evaluate your finetuned model on the `google/fleurs` test set using:

```bash
$ python test.py \
        --model_path ./whisper-tiny-hi-finetuned-run0-tiny \
        --model_size tiny \
        --language hi \
        --use_indic_norm
```

### Model Performance

- Baseline WER results on `google/fleurs -- hindi` (Indic Normalization)

| Model Size | Whisper Norm | Indic Norm | 
|------------|------------- |------------| 
| Tiny       | 172.60       | 196.57     | 
| Base       | 149.17       | 160.58     | 
| Small      | 67.37        | 89.73      |

- Fine-tuned Whisper WER Results on `google/fleurs -- hindi`:

| Model Size | Whisper Normalization | Indic Normalization |
|------------|-----------------------|---------------------|
| Tiny       | 14.21                 | 22.15              |
| Base       | 11.78                 | 19.44              |
| Small      | 10.11                 | 17.35              |

### Acknowledgments
We acknowledge the contributions of:

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the [Shrutilipi dataset](https://ai4bharat.iitm.ac.in/datasets/shrutilipi).
- [IIT Madras SpringLab](https://asr.iitm.ac.in/) for the [SpringX-Hindi dataset](https://asr.iitm.ac.in/dataset).
- [mozilla-foundation](https://foundation.mozilla.org/en/) for the [common_voice_11_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) hindi dataset.
- [google](https://huggingface.co/google) for the [fleurs](https://huggingface.co/datasets/google/fleurs) dataset.
- [IndicNLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library) for their powerful Indic normalization tools.

### Contributing
Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change. For any bugs or issues, please report them via the repository's issue tracker.

### License
This project is licensed under the MIT License.

### Citations

**Indic NLP LIbrary Citation**
```bibtex
@misc{kunchukuttan2020indicnlp,
  author = "Anoop Kunchukuttan",
  title = "{The IndicNLP Library}",
  year = "2020",
  howpublished={\url{https://github.com/anoopkunchukuttan/indic_nlp_library/blob/master/docs/indicnlp.pdf}}
}
```

**AI4Bharat - Shrutilipi dataset**
```bibtex
@misc{https://doi.org/10.48550/arxiv.2208.12666,
  doi = {10.48550/ARXIV.2208.12666},
  url = {https://arxiv.org/abs/2208.12666},
  author = {Bhogale, Kaushal Santosh and Raman, Abhigyan and Javed, Tahir and Doddapaneni, Sumanth and Kunchukuttan, Anoop and Kumar, Pratyush and Khapra, Mitesh M.},
  title = {Effectiveness of Mining Audio and Text Pairs from Public Data for Improving ASR Systems for Low-Resource Languages},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```