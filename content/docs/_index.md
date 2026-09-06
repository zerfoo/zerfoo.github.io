---
title: Documentation
weight: 1
bookToc: false
---

# Build AI into your Go application

Zerfoo brings model loading, generation, and serving into a Go library. Start with a compatible GGUF model, make your first inference call, and choose the interface that fits your application.

## Your first model

1. [Install Zerfoo](/docs/getting-started/installation/) with Go 1.26 or later.
2. [Follow the quickstart](/docs/getting-started/quickstart/) to load a local model and generate text.
3. [Check the model evidence](https://github.com/zerfoo/zerfoo/blob/main/docs/verified-models.md) before selecting a model and hardware configuration.

## Build the feature you need

- [Stream a response](/docs/cookbooks/streaming-chat/) as tokens arrive.
- [Generate structured JSON](/docs/cookbooks/structured-json-output/) with a schema.
- [Connect tools](/docs/cookbooks/tool-calling/) to model output.
- [Compute embeddings](/docs/cookbooks/embedding-similarity/) for similarity workflows.
- [Serve an HTTP API](/docs/tutorials/api-server/) for OpenAI-compatible clients.

## Understand the runtime

Read the [architecture overview](/docs/architecture/overview/), configure [GPU acceleration](/docs/architecture/gpu-setup/), or inspect the [benchmark conditions](/docs/reference/benchmarks/). Default and release builds are CGo-free; optional GPU build tags have additional requirements.

## Explore further

The framework also contains training, tabular ML, and time-series packages. Maturity varies by path: architecture registration and unit tests do not establish end-to-end verification. Consult the repository's evidence and known issues for your workload.

- [API reference](/docs/api/)
- [Cookbooks](/docs/cookbooks/)
- [Ecosystem modules](/docs/ecosystem/)
- [Model conversion](/docs/zonnx/)
- [Contributing](/docs/contributing/)
- [Go package reference](https://pkg.go.dev/github.com/zerfoo/zerfoo)
