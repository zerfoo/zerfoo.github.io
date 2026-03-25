---
title: zonnx Overview
weight: 1
bookToc: true
---

# zonnx Overview

zonnx is a standalone command-line tool that converts machine learning models to [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) format. It accepts ONNX and SafeTensors inputs and produces portable GGUF files compatible with both the zerfoo runtime and llama.cpp.

zonnx ships as a single static binary with no CGo dependency.

## Features

- **ONNX to GGUF conversion** -- convert decoder models (Llama, Gemma) from ONNX format
- **SafeTensors to GGUF conversion** -- convert encoder models (BERT, RoBERTa) from SafeTensors format
- **Post-conversion quantization** -- quantize weights to Q4_0 or Q8_0 during conversion
- **HuggingFace integration** -- download ONNX models and tokenizer files directly from the Hub
- **Model inspection** -- inspect ONNX and GGUF files for metadata, tensors, and structure
- **Architecture-aware mappings** -- tensor name and metadata mappings tuned per model family

## Installation

Requires Go 1.26 or later. Install with:

```bash
go install github.com/zerfoo/zonnx/cmd/zonnx@latest
```

Or build from source:

```bash
git clone https://github.com/zerfoo/zonnx.git
cd zonnx
go build -o zonnx ./cmd/zonnx
```

CGo is not required -- `CGO_ENABLED=0` works.

## Supported Architectures

| Architecture | `--arch` value | Input Formats | Notes |
|-------------|----------------|---------------|-------|
| Llama | `llama` (default) | ONNX | Llama 3, Code Llama |
| Gemma | `gemma` | ONNX | Gemma, Gemma 2, Gemma 3 |
| BERT | `bert` | ONNX, SafeTensors | Classification, embeddings |
| RoBERTa | `roberta` | ONNX, SafeTensors | Same layer structure as BERT |

Any architecture string can be passed via `--arch`. Generic metadata mapping applies to all architectures. Tensor name mapping currently covers Llama-style decoder models and BERT/RoBERTa encoder models.

## Basic Usage

```bash
# Download an ONNX model from HuggingFace
zonnx download --model google/gemma-2-2b-it --output ./models

# Convert ONNX to GGUF
zonnx convert --arch gemma --output ./models/model.gguf ./models/model.onnx

# Convert SafeTensors to GGUF
zonnx convert --format safetensors --arch bert --output ./models/model.gguf ./models/bert-dir/

# Convert with quantization
zonnx convert --quantize q4_0 --output ./models/model-q4.gguf ./models/model.onnx

# Inspect a model file
zonnx inspect --pretty ./models/model.onnx
```

## Commands

| Command | Description |
|---------|-------------|
| `convert` | Convert ONNX or SafeTensors models to GGUF |
| `download` | Download ONNX models and tokenizer files from HuggingFace Hub |
| `inspect` | Inspect ONNX or GGUF model files |

## Next Steps

- [ONNX to GGUF]({{< relref "onnx-to-gguf" >}}) -- step-by-step guide for converting ONNX models
- [SafeTensors to GGUF]({{< relref "safetensors-to-gguf" >}}) -- guide for converting SafeTensors models (BERT, RoBERTa)
