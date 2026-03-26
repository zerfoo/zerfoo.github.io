---
title: Your First Inference
weight: 3
bookToc: true
---

# Your First Inference

Go from zero to working LLM inference in under 5 minutes.

## Prerequisites

- **Go 1.25 or later** -- [download Go](https://go.dev/dl/)
- A machine with at least 4 GB of RAM (8 GB recommended for 7B models)
- Optional: an NVIDIA GPU with CUDA drivers for hardware-accelerated inference

Verify your Go installation:

```bash
go version
# go version go1.25.0 linux/amd64
```

## Install the CLI

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest
```

Verify:

```bash
zerfoo version
```

Zerfoo builds with zero CGo by default. GPU acceleration is loaded dynamically at runtime, so you do not need CUDA headers or build tags to compile.

## Download a Model

Zerfoo uses the GGUF model format -- the same format used by llama.cpp. Pull a small quantized model to get started:

```bash
zerfoo pull gemma-3-1b-q4
```

This downloads the GGUF file to `~/.cache/zerfoo`. You can also pull by full HuggingFace repo ID:

```bash
zerfoo pull meta-llama/Llama-3.2-1B-Instruct-GGUF
```

Manage cached models:

```bash
zerfoo list          # show cached models
zerfoo rm gemma-3-1b-q4  # remove a model
```

### Model aliases

Zerfoo ships with built-in aliases for popular models:

| Alias | HuggingFace Repo |
|-------|-----------------|
| `gemma-3-1b-q4` | `google/gemma-3-1b-it-qat-q4_0-gguf` |
| `llama-3-1b-q4` | `meta-llama/Llama-3.2-1B-Instruct-GGUF` |
| `llama-3-8b-q4` | `meta-llama/Llama-3.1-8B-Instruct-GGUF` |
| `mistral-7b-q4` | `mistralai/Mistral-7B-Instruct-v0.3-GGUF` |
| `qwen-2.5-7b-q4` | `Qwen/Qwen2.5-7B-Instruct-GGUF` |

You can also pass any HuggingFace repo ID directly, or a local file path.

## CLI Usage

### Interactive chat

Start a chat session with `zerfoo run`:

```bash
zerfoo run gemma-3-1b-q4
```

```text
Model loaded. Type your message (Ctrl-D to quit).

> What is the capital of France?
The capital of France is Paris.
>
```

### Single prompt

Run a one-off prompt with `predict`:

```bash
zerfoo predict --model gemma-3-1b-q4 --prompt "Explain what a tensor is in one paragraph."
```

### Sampling parameters

Both `run` and `predict` accept these flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--temperature` | Sampling temperature | 1.0 |
| `--top-k` | Top-K sampling | disabled |
| `--top-p` | Nucleus sampling | 1.0 |
| `--repetition-penalty` | Penalize repeated tokens | 1.0 |
| `--max-tokens` | Maximum tokens to generate | 256 |
| `--system` | System prompt | none |
| `--device` | Device (`cpu`, `cuda`) | `cpu` |

Example:

```bash
zerfoo predict \
  --model gemma-3-1b-q4 \
  --prompt "Write a haiku about Go." \
  --temperature 0.7 \
  --max-tokens 64
```

## Inference from Go Code

Zerfoo is designed to be embedded as a library. Create a new Go project:

```bash
mkdir my-llm-app && cd my-llm-app
go mod init my-llm-app
go get github.com/zerfoo/zerfoo@latest
```

Write `main.go`:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	// Load a quantized Gemma 3 1B model.
	// On first run, Zerfoo pulls the GGUF file from HuggingFace and caches it.
	mdl, err := inference.Load("gemma-3-1b-q4")
	if err != nil {
		log.Fatal(err)
	}
	defer mdl.Close()

	// Generate text from a prompt.
	result, err := mdl.Generate(
		context.Background(),
		"Explain what a tensor is in one paragraph.",
		inference.WithMaxTokens(128),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result)
}
```

Run it:

```bash
go run main.go
```

### Chat completion

For multi-turn conversations, use the `Chat` method:

```go
resp, err := mdl.Chat(context.Background(), []inference.Message{
	{Role: "system", Content: "You are a helpful assistant."},
	{Role: "user", Content: "What is the capital of France?"},
},
	inference.WithTemperature(0.5),
	inference.WithMaxTokens(64),
)
if err != nil {
	log.Fatal(err)
}
fmt.Println(resp.Content)
fmt.Printf("Tokens used: %d (prompt: %d, completion: %d)\n",
	resp.TokensUsed, resp.PromptTokens, resp.CompletionTokens)
```

### GPU acceleration

Pass `WithDevice` to run on a CUDA GPU:

```go
mdl, err := inference.LoadFile("model.gguf",
	inference.WithDevice("cuda"),
)
```

Or from the CLI:

```bash
zerfoo run gemma-3-1b-q4 --device cuda
```

No build tags are needed. Zerfoo discovers CUDA libraries at runtime. If CUDA is not available, the call returns an error so you can fall back to CPU gracefully.

## Serve an OpenAI-Compatible API

Start a server with `zerfoo serve`:

```bash
zerfoo serve gemma-3-1b-q4 --port 8080
```

Send a request with `curl`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-q4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

Enable streaming with SSE:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-q4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

Any OpenAI-compatible client library works -- just point it at `localhost:8080`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="gemma-3-1b-q4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Available endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion |
| POST | `/v1/completions` | Text completion |
| POST | `/v1/embeddings` | Text embeddings |
| GET | `/v1/models` | List loaded models |
| GET | `/metrics` | Prometheus metrics |

## Next Steps

- [Installation]({{< relref "/docs/getting-started/installation" >}}) -- detailed installation and platform support
