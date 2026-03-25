---
title: SafeTensors to GGUF
weight: 3
bookToc: true
---

# SafeTensors to GGUF Conversion

This guide covers converting SafeTensors models (typically BERT and RoBERTa) to GGUF format using zonnx. SafeTensors is HuggingFace's preferred serialization format for model weights.

## Prerequisites

- zonnx installed (`go install github.com/zerfoo/zonnx/cmd/zonnx@latest`)
- A HuggingFace model directory containing `config.json` and `model.safetensors`

## Directory Structure

zonnx expects a directory as input for SafeTensors conversion. The directory must contain:

```
model-dir/
  config.json           # required -- model configuration
  model.safetensors     # required -- model weights
```

The `config.json` provides architecture metadata (hidden size, layer count, attention heads, etc.) that zonnx maps to GGUF metadata keys. The `model.safetensors` file contains the weight tensors.

## Step 1: Download a Model

Download a model from HuggingFace. For example, to get [FinBERT](https://huggingface.co/ProsusAI/finbert) for financial sentiment analysis:

```bash
# Create a directory for the model
mkdir -p ./models/finbert

# Download config.json and model.safetensors
# (use the HuggingFace CLI, git clone, or manual download)
huggingface-cli download ProsusAI/finbert \
  --include config.json model.safetensors \
  --local-dir ./models/finbert
```

Verify the directory contents:

```bash
ls ./models/finbert/
# config.json  model.safetensors
```

## Step 2: Convert to GGUF

Run the `convert` command with `--format safetensors` and the appropriate `--arch`:

```bash
zonnx convert \
  --format safetensors \
  --arch bert \
  --output ./models/finbert.gguf \
  ./models/finbert/
```

Note that the input argument is the **directory** path, not the `.safetensors` file path.

## config.json Fields and Metadata Mapping

zonnx reads `config.json` and maps fields to GGUF metadata. For BERT and RoBERTa models, the following fields are mapped:

### Standard Fields (All Architectures)

| config.json field | GGUF key |
|-------------------|----------|
| `hidden_size` | `{arch}.embedding_length` |
| `num_hidden_layers` | `{arch}.block_count` |
| `num_attention_heads` | `{arch}.attention.head_count` |
| `num_key_value_heads` | `{arch}.attention.head_count_kv` |
| `intermediate_size` | `{arch}.feed_forward_length` |
| `vocab_size` | `{arch}.vocab_size` |
| `max_position_embeddings` | `{arch}.context_length` |

### BERT/RoBERTa-Specific Fields

| config.json field | GGUF key |
|-------------------|----------|
| `layer_norm_eps` | `{arch}.attention.layer_norm_epsilon` |
| `num_labels` | `{arch}.num_labels` |
| (auto) | `{arch}.pooler_type` = `"cls"` |

If `num_labels` is not present in `config.json` but `id2label` is, zonnx derives the label count from the `id2label` mapping.

## Supported Data Types

zonnx handles these SafeTensors data types:

| SafeTensors dtype | GGUF dtype |
|-------------------|------------|
| `F32` | Float32 |
| `F16` | Float16 |
| `BF16` | BFloat16 |

Non-float tensors (e.g., `position_ids` with int64 dtype) are skipped automatically during conversion.

## End-to-End Example: FinBERT

This example converts [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned for financial sentiment classification.

### 1. Download the Model

```bash
mkdir -p ./models/finbert
huggingface-cli download ProsusAI/finbert \
  --include config.json model.safetensors \
  --local-dir ./models/finbert
```

### 2. Inspect config.json

A typical FinBERT `config.json` contains:

```json
{
  "architectures": ["BertForSequenceClassification"],
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "intermediate_size": 3072,
  "vocab_size": 30522,
  "max_position_embeddings": 512,
  "layer_norm_eps": 1e-12,
  "id2label": {
    "0": "positive",
    "1": "negative",
    "2": "neutral"
  }
}
```

zonnx maps these fields to GGUF metadata keys like `bert.embedding_length`, `bert.block_count`, `bert.attention.head_count`, etc. The three labels in `id2label` produce `bert.num_labels = 3`.

### 3. Convert

```bash
zonnx convert \
  --format safetensors \
  --arch bert \
  --output ./models/finbert.gguf \
  ./models/finbert/
```

### 4. Verify

```bash
zonnx inspect --pretty ./models/finbert.gguf
```

The output should show GGUF metadata with `bert.*` keys and all encoder layer tensors.

### 5. Use with Zerfoo

```bash
zerfoo predict ./models/finbert.gguf --input "Revenue exceeded expectations this quarter"
```

## RoBERTa Models

RoBERTa conversion follows the same steps. Use `--arch roberta`:

```bash
zonnx convert \
  --format safetensors \
  --arch roberta \
  --output ./models/roberta.gguf \
  ./models/roberta-dir/
```

RoBERTa uses the same encoder layer structure as BERT. The `--arch` flag ensures tensor names are mapped using the `roberta.encoder.layer.N.*` prefix pattern.
