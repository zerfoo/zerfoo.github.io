---
title: "Granite Time Series"
weight: 2
bookToc: true
---

# Granite Time Series

Zerfoo supports IBM Granite Time Series foundation models for time-series
inference. Three model families are available, each targeting different tasks.

## Model Families

| Model | Parameters | Tasks | Key Feature |
|-------|-----------|-------|-------------|
| **Granite TTM** | 1M-5M | Forecasting | Zero-shot and few-shot forecasting with channel mixing |
| **Granite FlowState** | 2M-10M | Forecasting | Continuous forecasting across arbitrary timescales |
| **Granite TSPulse** | 1M-5M | Anomaly detection, classification, imputation, embedding | Lightweight encoder for multi-task time-series analysis |

## Supported Tasks

- **Forecasting** -- predict future values from historical context (TTM, FlowState)
- **Anomaly detection** -- identify outliers and anomalous patterns (TSPulse)
- **Classification** -- classify time-series segments (TSPulse)
- **Imputation** -- fill in missing values (TSPulse)
- **Embedding** -- extract fixed-size representations for downstream tasks (TSPulse)

## Converting Models

Granite Time Series models are published on HuggingFace in SafeTensors format.
Use the `granite2gguf` converter (part of `zonnx`) to produce GGUF files:

```bash
go install github.com/zerfoo/zonnx/cmd/granite2gguf@latest

# Convert a TTM model
granite2gguf \
  --model ibm-granite/granite-timeseries-ttm-r2 \
  --output granite-ttm-r2.gguf

# Convert a FlowState model
granite2gguf \
  --model ibm-granite/granite-timeseries-flowstate \
  --output granite-flowstate.gguf

# Convert a TSPulse model
granite2gguf \
  --model ibm-granite/granite-timeseries-tspulse \
  --output granite-tspulse.gguf
```

The converter downloads weights from HuggingFace, maps the architecture to GGUF
tensor names, and writes a self-contained `.gguf` file.

## Running Inference

### Forecasting (TTM)

```go
import "github.com/zerfoo/zerfoo/inference/timeseries"

model, err := timeseries.LoadGGUF("granite-ttm-r2.gguf", engine)
if err != nil {
    log.Fatal(err)
}
defer model.Close()

// Input: [batch, channels, context_length]
// Output: [batch, channels, forecast_length]
input := tensor.New[float32](engine, []int{1, 3, 512})
// ... fill input with historical data ...

forecast, err := model.Forecast(ctx, input)
if err != nil {
    log.Fatal(err)
}
fmt.Println("forecast shape:", forecast.Shape())
```

### Anomaly Detection (TSPulse)

```go
model, err := timeseries.LoadGGUF("granite-tspulse.gguf", engine)
if err != nil {
    log.Fatal(err)
}
defer model.Close()

scores, err := model.DetectAnomalies(ctx, input)
if err != nil {
    log.Fatal(err)
}
// scores: per-timestep anomaly scores
```

### Embedding Extraction (TSPulse)

```go
embeddings, err := model.Embed(ctx, input)
if err != nil {
    log.Fatal(err)
}
// embeddings: [batch, embed_dim] fixed-size representations
```

## Architecture Details

All three model families use a patch-based transformer encoder architecture:

1. **Patching** -- the input time series is segmented into fixed-size patches
2. **Channel mixing** -- multivariate channels are projected into a shared space
3. **Transformer encoder** -- standard multi-head self-attention over patches
4. **Task head** -- a linear projection head specific to the task (forecast, classify, reconstruct, embed)

GGUF metadata stores the model family (`granite-ttm`, `granite-flowstate`,
`granite-tspulse`), context length, forecast length, patch size, and number of
channels. The inference runtime auto-configures based on these fields.

## Model Sources

| Model | HuggingFace Repo |
|-------|-----------------|
| Granite TTM R2 | [ibm-granite/granite-timeseries-ttm-r2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) |
| Granite FlowState | [ibm-granite/granite-timeseries-flowstate](https://huggingface.co/ibm-granite/granite-timeseries-flowstate) |
| Granite TSPulse | [ibm-granite/granite-timeseries-tspulse](https://huggingface.co/ibm-granite/granite-timeseries-tspulse) |
