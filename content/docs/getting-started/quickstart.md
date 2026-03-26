---
title: Quick Start
weight: 2
bookToc: true
---

# Quick Start

Run your first LLM inference in under 5 minutes.

## Load a Model and Generate Text

Create a new Go project and add Zerfoo:

```bash
mkdir my-llm-app && cd my-llm-app
go mod init my-llm-app
go get github.com/zerfoo/zerfoo@latest
```

Write `main.go`:

```go
package main

import (
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo"
)

func main() {
	m, err := zerfoo.Load("google/gemma-3-4b")
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()

	reply, err := m.Chat("Explain quicksort in one sentence.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(reply)
}
```

Run it:

```bash
go run main.go
```

`zerfoo.Load` accepts a HuggingFace model ID (e.g. `"google/gemma-3-4b"`) or a local GGUF file path (e.g. `"./model.gguf"`). If the model is not cached locally it is downloaded automatically. The default quantization is Q4_K_M.

To request a specific quantization, append it to the ID:

```
google/gemma-3-4b/Q8_0
```

## Chat Completion

For multi-turn conversations, use the `Chat` method with structured messages:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	mdl, err := inference.Load("gemma-3-1b-q4")
	if err != nil {
		log.Fatal(err)
	}
	defer mdl.Close()

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
}
```

The `Chat` method formats messages using the model's built-in chat template and returns a `Response` with token usage statistics.

**CLI equivalent:**

```bash
zerfoo run gemma-3-1b-q4
```

This starts an interactive chat session:

```
Model loaded. Type your message (Ctrl-D to quit).

> What is the capital of France?
The capital of France is Paris.
>
```

## Stream Responses

Print tokens as they arrive:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo"
)

func main() {
	m, err := zerfoo.Load("google/gemma-3-4b")
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()

	stream, err := m.ChatStream(context.Background(), "Write a haiku about Go.")
	if err != nil {
		log.Fatal(err)
	}
	for tok := range stream {
		if tok.Done {
			break
		}
		fmt.Print(tok.Text)
	}
	fmt.Println()
}
```

For lower-level control, use the `inference` package directly:

```go
err = mdl.GenerateStream(ctx, "Tell me a joke.",
	generate.TokenStreamFunc(func(token string, done bool) error {
		if !done {
			fmt.Print(token)
		}
		return nil
	}),
	inference.WithMaxTokens(128),
)
```

**CLI equivalent:**

```bash
zerfoo predict --model gemma-3-1b-q4 --prompt "Write a haiku about Go."
```

## Generate Embeddings

Use the OpenAI-compatible API server to generate text embeddings. Start the server:

```bash
zerfoo serve gemma-3-1b-q4 --port 8080
```

Then request embeddings:

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-q4",
    "input": "Zerfoo is an ML framework for Go."
  }'
```

Any OpenAI-compatible client library works -- just point it at your server:

```go
// Using the standard net/http package
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
)

func main() {
	body, _ := json.Marshal(map[string]any{
		"model": "gemma-3-1b-q4",
		"input": "Zerfoo is an ML framework for Go.",
	})

	resp, err := http.Post(
		"http://localhost:8080/v1/embeddings",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()

	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)
	fmt.Println(result)
}
```

## Structured JSON Output

Control generation with temperature, token limits, and nucleus sampling to get structured output:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	mdl, err := inference.Load("gemma-3-1b-q4")
	if err != nil {
		log.Fatal(err)
	}
	defer mdl.Close()

	result, err := mdl.Generate(
		context.Background(),
		`Return a JSON object with the fields "name", "capital", and "population" for France. Output only valid JSON, no other text.`,
		inference.WithMaxTokens(128),
		inference.WithTemperature(0.0),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result)
}
```

**CLI equivalent:**

```bash
zerfoo predict \
  --model gemma-3-1b-q4 \
  --prompt 'Return a JSON object with the fields "name", "capital", and "population" for France. Output only valid JSON, no other text.' \
  --temperature 0 \
  --max-tokens 128
```

## Generation Options

Both the library API and CLI support these sampling parameters:

| Option | CLI Flag | Default | Description |
|--------|----------|---------|-------------|
| `WithTemperature` | `--temperature` | 1.0 | Sampling temperature |
| `WithTopP` | `--top-p` | 1.0 | Nucleus sampling |
| `WithTopK` | `--top-k` | disabled | Top-K sampling |
| `WithMaxTokens` | `--max-tokens` | 256 | Maximum tokens to generate |
| `WithRepetitionPenalty` | `--repetition-penalty` | 1.0 | Penalize repeated tokens |

Example with multiple options:

```go
result, err := m.Generate(context.Background(), "Tell me a joke.",
	zerfoo.WithGenTemperature(0.7),
	zerfoo.WithGenMaxTokens(128),
	zerfoo.WithGenTopP(0.9),
)
if err != nil {
	log.Fatal(err)
}
fmt.Println(result.Text)
fmt.Printf("Tokens: %d, Duration: %s\n", result.TokenCount, result.Duration)
```

## Next Steps

- [Installation](/docs/getting-started/installation) -- detailed installation and platform support
- [GPU Setup](/docs/architecture/gpu-setup) -- configure CUDA, ROCm, or OpenCL for hardware-accelerated inference
- [API Server](/docs/deployment) -- serve models behind an OpenAI-compatible HTTP API
- [API Reference](/docs/api) -- full API documentation
- [Tutorials](/docs/tutorials) -- step-by-step guides for common tasks
