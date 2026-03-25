---
title: Ecosystem
weight: 8
bookToc: true
bookCollapseSection: true
---

# Ecosystem

Zerfoo is a family of Go modules that together form a complete ML inference and training stack. Each module has its own `go.mod`, versioning, and release cycle.

## Dependency Graph

```
float16 ──┐
           ├──► ztensor ──► zerfoo
float8  ──┘                    ▲
                               │
ztoken ────────────────────────┘
```

- **float16** and **float8** provide reduced-precision arithmetic
- **ztensor** builds tensors, compute engines, and computation graphs on top of them
- **zerfoo** combines ztensor and ztoken into a full inference, training, and serving framework
- **ztoken** is independent (zero external deps) and plugs directly into zerfoo

## Which Module to Import

| You want to... | Import |
|----------------|--------|
| Run transformer inference or serve models | `github.com/zerfoo/zerfoo` |
| Work with tensors, GPU compute, or computation graphs | `github.com/zerfoo/ztensor` |
| Tokenize text (BPE, HuggingFace, GGUF) | `github.com/zerfoo/ztoken` |
| Do Float16 or BFloat16 arithmetic | `github.com/zerfoo/float16` |
| Do FP8 E4M3FN arithmetic | `github.com/zerfoo/float8` |
| Convert ONNX models to GGUF | `github.com/zerfoo/zonnx` (CLI) |

## Modules

### [ztensor]({{< relref "ztensor" >}})

GPU-accelerated tensor, compute engine, and computation graph library. Provides the `compute.Engine[T]` interface that powers all arithmetic in the ecosystem. Supports CUDA, ROCm, and OpenCL backends loaded at runtime via purego -- zero CGo.

### [ztoken]({{< relref "ztoken" >}})

BPE tokenizer with HuggingFace `tokenizer.json` and GGUF tokenizer extraction. Handles SentencePiece compatibility for Llama-family models. Zero external dependencies.

### [Numeric Types (float16 + float8)]({{< relref "numeric-types" >}})

IEEE 754 half-precision (`Float16`), Brain Floating Point (`BFloat16`), and FP8 E4M3FN (`Float8`) arithmetic libraries. Used by ztensor for quantized tensor storage and mixed-precision compute.

### zonnx

ONNX-to-GGUF converter CLI. Standalone binary with no runtime dependencies on the other modules. Converts ONNX models into GGUF format for use with zerfoo.
