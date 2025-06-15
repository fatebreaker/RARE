# RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models

This repository contains necessary scripts to run **RARE**.

[//]: # (> Link to paper: https://huggingface.co/papers/2408.06195, https://arxiv.org/abs/2408.06195 )

## Intro 

We propose **RARE**, Retrieval-Augmented Reasoning Enhancement for Large Language Models approach that significantly improves reasoning and factuality capabilities of large language models without fine-tuning or superior models.

<p align="center">
  <img src="assets/img.png">
</p>

## Prerequisites

- Python 3.10
- CUDA 12
- newest PyTorch
- newest `transformers`
- newest `vllm`

## Usage

### 1. Prepare corpus for retrieval:

```bash
cd MedRAG
# Extract individual corpora
python extract_chunks.py PubMed --db_dir ./corpus --output_dir ./chunks
python extract_chunks.py Textbooks --db_dir ./corpus --output_dir ./chunks
python extract_chunks.py StatPearls --db_dir ./corpus --output_dir ./chunks
python extract_chunks.py Wikipedia --db_dir ./corpus --output_dir ./chunks

# Or extract all medical corpora at once
python extract_chunks.py MedCorp --db_dir ./corpus --output_dir ./chunks
```

### 2. Build Retrieval Index
```bash
cd MedRAG
# Basic indexing
python colbert_indexer.py wikipedia ./chunks/Wikipedia_chunks.pickle

# Index all medical corpora
python colbert_indexer.py medcorp ./chunks/MedCorp_chunks.pickle

# Advanced indexing with custom settings
python colbert_indexer.py medcorp ./chunks/MedCorp_chunks.pickle \
  --checkpoint ./models/colbert_checkpoint \
  --output_dir ./indices \
  --doc_maxlen 512 \
  --nbits 4 \
  --nranks 2
 ```

### 3. Start ColBERT Search API
```bash
cd run_src
# Basic API server
python colbert_server.py ./colbert_indices/medcorp

# With collection file for enhanced results
python colbert_server.py ./colbert_indices/medcorp --collection ./chunks/MedCorp_chunks.pickle

# Custom host and port
python colbert_server.py ./colbert_indices/medcorp --host 127.0.0.1 --port 8080

# Production deployment
python colbert_server.py ./colbert_indices/medcorp --production --host 0.0.0.0 --port 80
 ```


### RARE

Here is an example to run RARE:

```bash
bash scripts/run_generate_medqa_8b.sh
```

The script `scripts/run_generate_medqa_8b.sh` includes several configurable parameters:
- `--dataset_name`: Name of the dataset (choose from [MATH, GSM8K, GSM8KHARD, STG, SVAMP, MULTIARITH]).
- `--test_json_filename`: Filename for the test JSON (default: test).
- `--model_ckpt`: Path to the model checkpoint.
- `--retrieval_corpus`: retrieval corpus that can be use to retrieve information.
- `--num_rollouts`: Number of rollouts (default: 4).

Make sure to adjust these parameters according to your requirements.