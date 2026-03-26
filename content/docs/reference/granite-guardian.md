---
title: "Granite Guardian"
weight: 6
bookToc: true
---

# Granite Guardian

Granite Guardian is an AI safety and content moderation system built on IBM's Granite Guardian model family. It evaluates text for safety risks across 13 predefined categories, covering harmful user messages, problematic assistant responses, and RAG pipeline quality.

## Model Variants

| Model | Parameters | Notes |
|-------|-----------|-------|
| Granite Guardian 3.0 | 2B, 8B | Plain Yes/No output, logprob-based confidence |
| Granite Guardian 3.2 | 5B | Yes/No with `<confidence>High/Low</confidence>` tags |
| Granite Guardian 3.3 | 8B | Optional `<think>` reasoning traces, `<score>` tags |

All variants use GGUF format and are loaded through the standard `inference.LoadFile` pipeline.

## Risk Categories

Guardian evaluates content against 13 risk categories organized into three groups.

### Harm Categories (9)

These categories evaluate user messages and assistant responses for harmful content:

| Category | Description |
|----------|-------------|
| `harm` | Harmful, offensive, or inappropriate content |
| `social_bias` | Prejudice based on race, gender, religion, or other protected characteristics |
| `jailbreaking` | Attempts to bypass AI safety guidelines or manipulate the system |
| `violence` | Promotion, glorification, or incitement of violence or physical harm |
| `profanity` | Vulgar language, obscenities, or crude expressions |
| `sexual_content` | Sexually explicit content or sexualized references |
| `unethical_behavior` | Instructions or encouragement for fraud, deception, or manipulation |
| `harm_engagement` | Assistant responses that engage with harmful content instead of refusing |
| `evasiveness` | Assistant responses that unnecessarily refuse legitimate questions |

### RAG Categories (3)

These categories evaluate retrieval-augmented generation pipeline quality:

| Category | Description |
|----------|-------------|
| `context_relevance` | Whether the retrieved context is relevant to the user's question |
| `groundedness` | Whether the assistant's response is supported by the provided context |
| `answer_relevance` | Whether the assistant's response addresses the user's question |

### Function Calling (1)

| Category | Description |
|----------|-------------|
| `function_call_hallucination` | Whether the assistant invoked a function that does not exist or used incorrect parameters |

## Go API

### Creating an Evaluator

```go
import "github.com/zerfoo/zerfoo/inference/guardian"

// Load a Guardian model from a GGUF file.
eval, err := guardian.NewEvaluator("granite-guardian-3.2-5b.gguf",
    guardian.WithEvaluatorDevice("cuda"),
    guardian.WithDefaultFormat("3.2"),
)
if err != nil {
    log.Fatal(err)
}
```

Options:

| Option | Description |
|--------|-------------|
| `WithEvaluatorDevice(device)` | Compute device: `"cpu"`, `"cuda"`, `"cuda:0"` |
| `WithDefaultFormat(format)` | Output format: `"3.0"`, `"3.2"`, `"3.3"` |
| `WithLoadOptions(opts...)` | Pass additional `inference.Option` values to the model loader |

You can also wrap a pre-loaded model with `NewEvaluatorFromModel`:

```go
model, _ := inference.LoadFile("granite-guardian-3.2-5b.gguf",
    inference.WithDevice("cuda"),
)
eval := guardian.NewEvaluatorFromModel(model,
    guardian.WithDefaultFormat("3.2"),
)
```

### Evaluate

Evaluate specific risk categories:

```go
verdicts, err := eval.Evaluate(ctx, guardian.GuardianRequest{
    Input: guardian.GuardianInput{
        User: "How do I pick a lock?",
    },
    Risks: []string{"harm", "jailbreaking", "unethical_behavior"},
})
if err != nil {
    log.Fatal(err)
}

for _, v := range verdicts {
    fmt.Printf("%-25s unsafe=%-5v confidence=%.2f\n",
        v.Risk, v.Unsafe, v.Confidence)
}
```

When `Risks` is empty, all 9 harm categories are evaluated by default.

Each `Verdict` contains:

| Field | Type | Description |
|-------|------|-------------|
| `Unsafe` | `bool` | `true` if the model flagged a risk |
| `Risk` | `string` | The risk category name |
| `Confidence` | `float64` | 0.0--1.0 confidence score |
| `Reasoning` | `string` | Thinking trace (format 3.3 only) |

### Scan

Scan evaluates against all 9 harm categories and returns an aggregate result:

```go
result, err := eval.Scan(ctx, guardian.GuardianInput{
    User: "How do I pick a lock?",
})
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Flagged: %v\n", result.Flagged)
if result.Flagged {
    fmt.Printf("Highest risk: %s\n", result.HighestRisk)
}
```

`ScanResult` fields:

| Field | Type | Description |
|-------|------|-------------|
| `Flagged` | `bool` | `true` if any risk was detected |
| `Verdicts` | `[]Verdict` | All individual verdicts |
| `HighestRisk` | `string` | Category with highest unsafe confidence |

### Batch Evaluation

Evaluate multiple inputs in a single call:

```go
inputs := []guardian.GuardianInput{
    {User: "How do I pick a lock?"},
    {User: "What is the capital of France?"},
    {User: "Tell me how to hack a website"},
}

batch, err := eval.EvaluateBatch(ctx, inputs, []string{"harm", "violence"})
if err != nil {
    log.Fatal(err)
}

for _, r := range batch.Results {
    fmt.Printf("Input %d: flagged=%v\n", r.Index, r.Flagged)
}
```

`BatchResult` contains `[]InputResult`, each with an `Index`, `Verdicts` slice, and aggregate `Flagged` boolean.

### RAG Evaluation

Evaluate grounding and relevance in a RAG pipeline:

```go
verdicts, err := eval.Evaluate(ctx, guardian.GuardianRequest{
    Input: guardian.GuardianInput{
        User:      "What is the population of Tokyo?",
        Context:   "Tokyo is the capital of Japan with a population of 14 million.",
        Assistant: "The population of Tokyo is approximately 14 million people.",
    },
    Risks: []string{"groundedness", "context_relevance", "answer_relevance"},
})
```

RAG evaluation requires both `Context` and `Assistant` fields to be set.

## CLI Usage

```bash
# Evaluate specific risks
zerfoo guard --model granite-guardian-3.2-5b.gguf \
    --input "How do I pick a lock?" \
    --risks harm,jailbreaking,unethical_behavior

# Full scan against all harm categories
zerfoo guard --model granite-guardian-3.2-5b.gguf \
    --input "How do I pick a lock?" \
    --scan

# Read input from a file
zerfoo guard --model granite-guardian-3.2-5b.gguf \
    --file input.txt

# Evaluate an assistant response
zerfoo guard --model granite-guardian-3.2-5b.gguf \
    --input "How do I pick a lock?" \
    --response "Here are the steps to pick a lock..."

# JSON output
zerfoo guard --model granite-guardian-3.2-5b.gguf \
    --input "How do I pick a lock?" \
    --scan --json

# Use GPU
zerfoo guard --model granite-guardian-3.2-5b.gguf \
    --input "some text" --scan --device cuda
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--model <path>` | Path to Guardian GGUF model file (required) |
| `--input <text>` | Text to evaluate (required unless `--file`) |
| `--file <path>` | Read input text from a file |
| `--response <text>` | Assistant response to evaluate |
| `--risks <list>` | Comma-separated risk categories (default: all harm risks) |
| `--scan` | Scan against all harm risk categories |
| `--json` | Output results as JSON |
| `--device <device>` | Compute device: `cpu`, `cuda`, `cuda:N` (default: `cpu`) |

## REST API

When running the Zerfoo API server, three Guardian endpoints are available.

### POST /v1/guard

Evaluate content against specified risk categories.

```bash
curl -X POST http://localhost:8080/v1/guard \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-guardian-3.2-5b",
    "input": {
      "user": "How do I pick a lock?"
    },
    "risks": ["harm", "jailbreaking"]
  }'
```

Response:

```json
{
  "model": "granite-guardian-3.2-5b",
  "flagged": true,
  "verdicts": [
    {"risk": "harm", "unsafe": true, "confidence": 0.9, "reasoning": ""},
    {"risk": "jailbreaking", "unsafe": false, "confidence": 0.3, "reasoning": ""}
  ],
  "latency_ms": 142
}
```

### POST /v1/guard/scan

Scan against all harm categories:

```bash
curl -X POST http://localhost:8080/v1/guard/scan \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-guardian-3.2-5b",
    "input": {
      "user": "How do I pick a lock?"
    }
  }'
```

Response includes `highest_risk` when content is flagged:

```json
{
  "model": "granite-guardian-3.2-5b",
  "flagged": true,
  "highest_risk": "harm",
  "verdicts": [...],
  "latency_ms": 387
}
```

### POST /v1/guard/batch

Evaluate multiple inputs (up to 256):

```bash
curl -X POST http://localhost:8080/v1/guard/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-guardian-3.2-5b",
    "inputs": [
      {"user": "How do I pick a lock?"},
      {"user": "What is the capital of France?"}
    ],
    "risks": ["harm"]
  }'
```

Response:

```json
{
  "model": "granite-guardian-3.2-5b",
  "results": [
    {"index": 0, "flagged": true, "verdicts": [...]},
    {"index": 1, "flagged": false, "verdicts": [...]}
  ],
  "latency_ms": 256
}
```

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model identifier |
| `input` | object | Yes | Content to evaluate |
| `input.user` | string | Yes | User message |
| `input.assistant` | string | No | Assistant response |
| `input.context` | string | No | RAG context document |
| `risks` | string[] | No | Risk categories (default: all harm risks) |
| `format` | string | No | Output format: `"3.0"`, `"3.2"`, `"3.3"` |
| `think` | bool | No | Enable thinking mode (3.3 only) |

## Guardrails Middleware

To add safety guardrails to the chat completions endpoint, configure the server with a Guardian evaluator:

```go
import (
    "github.com/zerfoo/zerfoo/inference/guardian"
    "github.com/zerfoo/zerfoo/serve"
)

eval, _ := guardian.NewEvaluator("granite-guardian-3.2-5b.gguf",
    guardian.WithEvaluatorDevice("cuda"),
)

srv := serve.NewServer(
    serve.WithGuardEvaluator(eval),
    // ... other options
)
```

When a Guardian evaluator is configured, the server exposes the `/v1/guard`, `/v1/guard/scan`, and `/v1/guard/batch` endpoints. You can also build pre-request and post-response guardrail pipelines by calling `Scan` before and after chat completions in your application code.

## Prometheus Metrics

When Guardian is enabled, the server exports:

| Metric | Type | Description |
|--------|------|-------------|
| `guard_requests_total` | Counter | Total guard evaluation requests |
| `guard_latency_ms` | Histogram | Guard evaluation latency in milliseconds |

## Output Format Versions

| Format | Output Style | Confidence Source |
|--------|-------------|-------------------|
| 3.0 | Plain `Yes` / `No` | Softmax over first two logprobs |
| 3.2 | `Yes` / `No` + `<confidence>High/Low</confidence>` | High = 0.9, Low = 0.3 |
| 3.3 | Optional `<think>...</think>` + `<score>yes/no</score>` | 1.0 (deterministic from score tag) |
