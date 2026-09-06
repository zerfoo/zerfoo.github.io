---
title: Benchmarks
weight: 1
bookToc: true
---

# Performance, with the conditions attached

Zerfoo publishes measurements so you can assess the runtime against your own workload. These results describe one machine and configuration. They are not independently reproduced or quality-normalized.

## Recorded comparison: March 27, 2026

| Model | GGUF quantization | Zerfoo tok/s | Ollama tok/s | Ratio |
|---|---|---:|---:|---:|
| Gemma 3 1B | Q4_K_M | 235 | 188 | 1.25× |
| DeepSeek-R1-Distill 1.5B | Q4_K_M | 186 | 167 | 1.11× |
| Llama 3.2 3B | Q4_K_M | 92 | 93 | 0.99× |
| Mistral 7B | Q5_K_M | 44 | 44 | 1.00× |

The measured advantage is concentrated in the smaller models. At 3B and 7B these results are approximately at parity.

### Setup

- NVIDIA DGX Spark GB10, Grace Blackwell, sm_121; 128 GB LPDDR5x unified memory.
- Fixed 8-word prompt: “Explain the theory of relativity in simple terms.”
- 128 generated tokens, greedy sampling, batch size 1.
- Three-run median; fp32 compute and KV cache.
- Ollama 0.17.7.

### What affects the comparison

Zerfoo's timer spans prefill and decode. Ollama's eval rate covers decode only. Quantization labels describe the file: Zerfoo re-quantizes Q4_K weights to Q4_0 at load, while Ollama uses Q4_K_M arithmetic. No perplexity comparison was performed, and Ollama model tags were unpinned.

One prompt and one token count do not characterize long-context, batched, or concurrent serving. These results are specific to GB10 unified memory and should not be generalized to other hardware. Ollama is a convenience baseline, not a measurement of the performance ceiling across inference engines.

[Read the raw results](https://github.com/zerfoo/zerfoo/blob/main/results/benchmark-2026-03-27.json) and [the repository's full methodology](https://github.com/zerfoo/zerfoo/blob/main/README.md#benchmarks). The repository is the source of record for reproduction instructions and updated evidence.

## Models larger than memory

A single CPU-only run on March 29, 2026 loaded a MiniMax-M2 229B model with 128.8 GB of Q4_K_M weights across three shards on a 128 GB machine. Loading took 6.3 seconds; generation reached 0.06 tok/s for four tokens.

This demonstrates that memory-mapped loading can make an over-RAM model accessible. It is not an interactive-throughput result or an output-quality validation. Weight paging is NVMe-bound, and there is no GPU path for over-RAM inference in this recorded setup.

## Choose by evidence

A registered architecture means the graph builder exists. It does not mean every model, quantization, or generation path has been validated. Consult the [verified-model matrix](https://github.com/zerfoo/zerfoo/blob/main/docs/verified-models.md) and [open issues](https://github.com/zerfoo/zerfoo/issues) when evaluating your workload.
