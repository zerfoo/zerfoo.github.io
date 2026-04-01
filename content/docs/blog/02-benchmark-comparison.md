---
title: "Zerfoo vs Ollama vs llama.cpp: A Performance Comparison"
weight: 2
bookToc: true
---

# Zerfoo vs Ollama vs llama.cpp: A Performance Comparison

> **Update 2026-03-27:** Benchmarks updated to multi-model 3-run median methodology. Gemma 3 1B: 241 tok/s (Ollama 188 tok/s) = 25% faster. Additional models: DeepSeek R1 1.5B (186 vs 167, +11%), Llama 3.2 3B (92 vs 93, parity), Mistral 7B (44 vs 44, parity).

When we set out to build an ML inference framework in Go, the first question everyone asked was: "Can Go actually compete with C++ on inference throughput?" The answer is yes. On Gemma 3 1B Q4_K_M, Zerfoo decodes at **241 tokens/second** — 28% faster than Ollama on the same NVIDIA DGX Spark hardware.

This post breaks down how we measured these numbers, what architectural decisions make them possible, and how you can reproduce the results on your own hardware.

## The Numbers

All measurements use the same GGUF model file, the same prompt ("The meaning of life is"), and measure steady-state decode throughput after warm-up on an NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified LPDDR5x, CUDA 13.0).

| Model | Zerfoo (tok/s) | Ollama (tok/s) | Speedup |
|-------|----------------|----------------|---------|
| **Gemma 3 1B Q4_K_M** | **241** | 188 | **+28%** |
| DeepSeek R1 1.5B | 186 | 167 | +11% |
| Llama 3.2 3B | 92 | 93 | parity |
| Mistral 7B | 44 | 44 | parity |

All numbers are 3-run medians from the 2026-03-27 multi-model benchmark. The gap narrows at larger model sizes where memory bandwidth dominates over kernel launch overhead. On smaller models where kernel fusion matters most, Zerfoo's CUDA graph capture and fused kernels provide a clear advantage.

## Why Zerfoo Is Faster

### 1. CUDA Graph Capture (99.5% Coverage)

The single biggest performance win. During the first decode step, Zerfoo captures the entire forward pass — 26 transformer layers, attention, FFN, normalization — as a single CUDA graph. Every subsequent decode step replays that graph in one GPU launch instead of dispatching hundreds of individual kernel launches.

Each kernel launch costs 5-10 microseconds of CPU-GPU synchronization overhead. With hundreds of kernels per token, that adds up to milliseconds of wasted time per token. CUDA graph capture eliminates this entirely.

Zerfoo achieves 99.5% instruction coverage in CUDA graph capture on the GGUF inference path. The remaining 0.5% consists of operations that cannot be captured (host-side sampling, tokenizer lookup).

### 2. Fused Kernels

Operations that are separate kernel launches in other frameworks are fused into single kernels in Zerfoo:

- **FusedAddRMSNorm** — Residual addition and RMS normalization in a single memory pass. Instead of reading the hidden state twice (once for add, once for norm), we do it once.

- **FusedQKNormRoPE** — Query/Key normalization and rotary position embeddings combined. This eliminates an intermediate buffer and a kernel launch.

- **FusedSiluGate** — SiLU activation and gating in the FFN, fused into one kernel.

- **Merged QKV and Gate+Up projections** — A single GEMV call replaces 2-3 separate matrix-vector multiplies.

Each fusion eliminates a kernel launch (5-10 us saved), a memory round-trip (reading/writing the full hidden state), and an intermediate buffer allocation.

### 3. Zero CGo Overhead

Most Go programs that call into C libraries use CGo, which adds approximately 200 nanoseconds of overhead per call for the goroutine stack switch. Zerfoo uses purego (dlopen at runtime) to call CUDA APIs directly, bypassing CGo entirely.

This matters because a single decode step involves thousands of CUDA API calls — memory copies, kernel launches, synchronization points. At 200ns per call, CGo overhead alone would cost hundreds of microseconds per token.

### 4. Optimized Q4_0 GEMV

The quantized matrix-vector multiply kernel — the innermost loop of decode — is hand-tuned with coalesced memory access patterns and efficient warp-level reductions. Since decode processes one token at a time (GEMV, not GEMM), the memory access pattern is critical, and our kernel is optimized specifically for this case.

## Methodology

### Zerfoo

```bash
git clone https://github.com/zerfoo/zerfoo.git
cd zerfoo

# Place gemma-3-1b-it-Q4_K_M.gguf in models/
go run ./cmd/bench \
  --model models/gemma-3-1b-it-Q4_K_M.gguf \
  --tokens 256 \
  --warmup 3 \
  --prompt "The meaning of life is"
```

The `cmd/bench` harness reports throughput (tok/s), time-to-first-token (TTFT), P99 latency, and GPU memory usage. CUDA graph capture is enabled by default on supported GPUs.

### Ollama

```bash
ollama pull gemma3:1b
ollama run gemma3:1b "The meaning of life is" --verbose
```

Look for `eval rate: XXX.XX tokens/s` in the verbose output.

### llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

./build/bin/llama-bench \
  -m /path/to/gemma-3-1b-it-Q4_K_M.gguf \
  -p 0 -n 256 -ngl 99
```

The `-p 0` flag skips prompt processing to measure pure decode throughput. `-ngl 99` offloads all layers to GPU.

## Fair Comparison Guidelines

If you want to run your own benchmarks, here's how to keep the comparison fair:

1. **Same model file.** All three frameworks read GGUF. Use the exact same `.gguf` file for each run.

2. **Match token counts.** Generate the same number of tokens (256 is a good default).

3. **Warm up.** Run at least 3 warm-up iterations. Zerfoo's `cmd/bench` handles this with `--warmup 3`.

4. **Isolate the GPU.** Close other GPU workloads. Check with `nvidia-smi` that no other processes are using the GPU.

5. **Measure decode throughput.** All numbers in this post are decode throughput (tokens per second during autoregressive generation), not prompt processing (prefill) speed. These are fundamentally different workloads — prefill is compute-bound (GEMM), decode is memory-bandwidth-bound (GEMV).

6. **Record your environment.** Report GPU model, CUDA version, driver version, CPU, RAM, OS, and framework version/commit hash.

## What About Other GPUs?

We've measured on the DGX Spark so far. We expect similar relative performance on other NVIDIA GPUs, but absolute numbers will vary with memory bandwidth and compute capability. We welcome community benchmark contributions:

| GPU | Zerfoo (est.) | Status |
|-----|---------------|--------|
| DGX Spark GB10 | 241 tok/s | Measured (3-run median, 2026-03-27) |
| RTX 4090 | TBD | Community contributions welcome |
| RTX 3090 | TBD | Community contributions welcome |
| A100 80GB | TBD | Community contributions welcome |
| Apple M-series (CPU) | ~8-15 tok/s | Metal backend not yet implemented |

If you run benchmarks on your hardware, we'd love to include your results. Open an issue or PR with your numbers, methodology, and environment details.

## The Bottom Line

A Go inference framework can match and exceed C++ runtimes on decode throughput. The key insight is that modern inference is GPU-bound, not language-bound — what matters is how efficiently you use the GPU, not what language your host code is written in. CUDA graph capture, fused kernels, and zero CGo overhead let Zerfoo spend its time where it matters: on the GPU.

Try it yourself:

```bash
go get github.com/zerfoo/zerfoo@latest
```
