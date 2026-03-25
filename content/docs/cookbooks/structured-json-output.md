---
title: Structured JSON Output
weight: 6
bookToc: true
---

# Structured JSON Output

Use grammar-guided decoding to guarantee the model outputs valid JSON that conforms to a schema. This eliminates the need for fragile regex parsing or retry loops when you need structured data from a language model.

Two approaches are shown:

1. **High-level** -- `zerfoo.WithSchema` (recommended for most use cases)
2. **Low-level** -- `grammar.Convert` + `inference.WithGrammar` (for custom pipelines)

## Usage

```bash
go run ./docs/cookbook/06-structured-json-output/ --model path/to/model.gguf
go run ./docs/cookbook/06-structured-json-output/ --model path/to/model.gguf --low-level
```

## Full Code

```go
// Recipe 06: Structured JSON Output
//
// Use grammar-guided decoding to guarantee the model outputs valid JSON that
// conforms to a schema. This eliminates the need for fragile regex parsing
// or retry loops when you need structured data from a language model.
//
// Two approaches are shown:
//  1. High-level: zerfoo.WithSchema (recommended for most use cases)
//  2. Low-level: grammar.Convert + inference.WithGrammar (for custom pipelines)
//
// Usage:
//
//	go run ./docs/cookbook/06-structured-json-output/ --model path/to/model.gguf
//	go run ./docs/cookbook/06-structured-json-output/ --model path/to/model.gguf --low-level
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo"
	"github.com/zerfoo/zerfoo/generate/grammar"
	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file or HuggingFace model ID")
	prompt := flag.String("prompt", "Generate a JSON object for a city: Paris, population 2.1 million, country France.", "generation prompt")
	lowLevel := flag.Bool("low-level", false, "use low-level grammar.Convert + inference.WithGrammar")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: structured-json-output --model <path> [--low-level]")
		os.Exit(1)
	}

	// Define the JSON schema. The model's output is constrained to this shape.
	schema := grammar.JSONSchema{
		Type: "object",
		Properties: map[string]*grammar.JSONSchema{
			"name":       {Type: "string"},
			"population": {Type: "number"},
			"country":    {Type: "string"},
		},
		Required: []string{"name", "population", "country"},
	}

	if *lowLevel {
		runLowLevel(*modelPath, *prompt, &schema)
	} else {
		runHighLevel(*modelPath, *prompt, schema)
	}
}

// runHighLevel uses the one-line zerfoo.WithSchema option.
func runHighLevel(modelPath, prompt string, schema grammar.JSONSchema) {
	m, err := zerfoo.Load(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	result, err := m.Generate(context.Background(), prompt,
		zerfoo.WithSchema(schema),
		zerfoo.WithGenMaxTokens(128),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(result.Text)
}

// runLowLevel converts the schema to a grammar and passes it to the inference
// package directly. Use this when you need full control over the pipeline.
func runLowLevel(modelPath, prompt string, schema *grammar.JSONSchema) {
	// Convert JSON schema to a grammar state machine.
	g, err := grammar.Convert(schema)
	if err != nil {
		fmt.Fprintf(os.Stderr, "grammar convert: %v\n", err)
		os.Exit(1)
	}

	model, err := inference.LoadFile(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	text, err := model.Generate(context.Background(), prompt,
		inference.WithMaxTokens(128),
		inference.WithGrammar(g),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(text)
}
```

## How It Works

### High-Level Approach

`zerfoo.WithSchema` is the simplest way to get structured output. Pass a `grammar.JSONSchema` and the framework handles schema-to-grammar conversion and constrained decoding automatically. Every token the model generates is validated against the schema, so the output is guaranteed to be valid JSON with the specified fields.

### Low-Level Approach

For custom pipelines, the `grammar` package exposes the conversion step directly:

1. **Schema definition** -- Define a `grammar.JSONSchema` with the expected type, properties, and required fields.
2. **Grammar conversion** -- `grammar.Convert` compiles the JSON schema into a grammar state machine that constrains which tokens can appear at each position.
3. **Constrained decoding** -- `inference.WithGrammar` passes the compiled grammar to the decoder, which masks out invalid tokens at each step.

### When to Use Each

| Approach | Use case |
|----------|----------|
| `zerfoo.WithSchema` | Most applications -- simple, one-line integration |
| `grammar.Convert` + `inference.WithGrammar` | Custom inference pipelines, reusing compiled grammars across requests, or combining with other inference options |

## See Also

- [Basic Text Generation](/docs/cookbooks/basic-text-generation) -- unconstrained text generation
- [Custom Sampling](/docs/cookbooks/custom-sampling) -- control randomness with temperature, top-K, and top-P
