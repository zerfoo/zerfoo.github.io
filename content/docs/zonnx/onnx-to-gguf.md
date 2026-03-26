---
title: ONNX to GGUF
weight: 2
bookToc: true
---

# ONNX to GGUF Conversion

This guide walks through converting an ONNX model to GGUF format using zonnx. The resulting GGUF file can be loaded by zerfoo or llama.cpp.

## Prerequisites

- zonnx installed (`go install github.com/zerfoo/zonnx/cmd/zonnx@latest`)
- An ONNX model file, either local or on HuggingFace

## Step 1: Download a Model from HuggingFace

Use the `download` command to fetch an ONNX model and its tokenizer files:

```bash
zonnx download --model google/gemma-2-2b-it --output ./models
```

For gated models that require authentication:

```bash
# Via flag
zonnx download --model meta-llama/Llama-3-8B --output ./models --api-key YOUR_HF_TOKEN

# Via environment variable
export HF_API_KEY=YOUR_HF_TOKEN
zonnx download --model meta-llama/Llama-3-8B --output ./models
```

The `--api-key` flag takes precedence over the `HF_API_KEY` environment variable.

After downloading, you should have at minimum:

```text
models/
  model.onnx
  config.json        # optional but recommended for metadata
  tokenizer.json     # downloaded automatically if available
```

## Step 2: Convert to GGUF

Run the `convert` command with the appropriate `--arch` flag:

```bash
zonnx convert --arch gemma --output ./models/gemma-2b.gguf ./models/model.onnx
```

### The `--arch` Flag

The `--arch` flag selects the tensor name mapping and metadata mapping for the target architecture. If a `config.json` file exists alongside the ONNX file, zonnx reads it automatically and maps HuggingFace config fields to GGUF metadata keys.

If `--arch` is omitted, it defaults to `llama`.

### Convert Command Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `<input-dir>/<input-base>.gguf` | Output GGUF file path |
| `--arch` | `llama` | Model architecture for metadata and tensor mapping |
| `--format` | `onnx` | Input format: `onnx` or `safetensors` |
| `--quantize` | (none) | Quantize weights: `q4_0` or `q8_0` |

## Step 3: Quantize During Conversion (Optional)

To reduce model size, quantize weights during conversion:

```bash
# 4-bit quantization (smallest, some quality loss)
zonnx convert --arch gemma --quantize q4_0 --output ./models/gemma-2b-q4.gguf ./models/model.onnx

# 8-bit quantization (good balance of size and quality)
zonnx convert --arch gemma --quantize q8_0 --output ./models/gemma-2b-q8.gguf ./models/model.onnx
```

| Quantization | Bits per Weight | Use Case |
|-------------|-----------------|----------|
| (none) | 32 | Full precision, largest file |
| `q8_0` | 8 | Good quality, ~4x smaller than F32 |
| `q4_0` | 4 | Smallest, ~8x smaller than F32 |

## Step 4: Verify the Output

Inspect the generated GGUF file to confirm metadata and tensors:

```bash
zonnx inspect --pretty ./models/gemma-2b.gguf
```

## Supported Architectures

| Architecture | `--arch` value | Tensor Mapping | Notes |
|-------------|----------------|----------------|-------|
| Llama | `llama` (default) | Decoder layers (`model.layers.N.*`) | Llama 3, Code Llama |
| Gemma | `gemma` | Decoder layers (`model.layers.N.*`) | Gemma, Gemma 2, Gemma 3 |
| BERT | `bert` | Encoder layers (`bert.encoder.layer.N.*`) | Classification, embeddings |
| RoBERTa | `roberta` | Encoder layers (`roberta.encoder.layer.N.*`) | Same structure as BERT |

## Metadata Mapping

When a `config.json` file is present alongside the ONNX model, zonnx maps these HuggingFace fields to GGUF metadata:

| config.json field | GGUF key |
|-------------------|----------|
| `hidden_size` | `{arch}.embedding_length` |
| `num_hidden_layers` | `{arch}.block_count` |
| `num_attention_heads` | `{arch}.attention.head_count` |
| `num_key_value_heads` | `{arch}.attention.head_count_kv` |
| `intermediate_size` | `{arch}.feed_forward_length` |
| `vocab_size` | `{arch}.vocab_size` |
| `max_position_embeddings` | `{arch}.context_length` |
| `rms_norm_eps` | `{arch}.attention.layer_norm_rms_epsilon` |
| `rope_theta` | `{arch}.rope.freq_base` |

## Using the GGUF File with Zerfoo

Once converted, load the model with zerfoo:

```bash
zerfoo run ./models/gemma-2b.gguf --prompt "Hello, world!"
```

Or serve it as an OpenAI-compatible API:

```bash
zerfoo serve ./models/gemma-2b.gguf
```
