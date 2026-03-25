---
title: Batch Inference
weight: 8
bookToc: true
---

# Batch Inference

Run inference over many prompts concurrently using goroutines. This pattern is useful for processing datasets, evaluations, or any batch workload where you need to generate completions for a list of inputs.

The program loads a single model and fans out generation across a configurable number of worker goroutines, collecting results in order.

## Full Example

```go
// Recipe 08: Batch Inference
//
// Run inference over many prompts concurrently using goroutines. This pattern
// is useful for processing datasets, evaluations, or any batch workload.
//
// The program loads a single model and fans out generation across a configurable
// number of worker goroutines, collecting results in order.
//
// Usage:
//
//	go run ./docs/cookbook/08-batch-inference/ --model path/to/model.gguf
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"sync"

	"github.com/zerfoo/zerfoo"
)

// prompts is a batch of inputs to process.
var prompts = []string{
	"Summarize the Go memory model in one sentence.",
	"What is a goroutine?",
	"Explain channels in Go.",
	"What is the purpose of the context package?",
	"Describe Go's approach to error handling.",
	"What is a defer statement?",
	"Explain interfaces in Go.",
	"What are Go modules?",
}

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file or HuggingFace model ID")
	workers := flag.Int("workers", 4, "number of concurrent workers")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: batch-inference --model <path> [--workers 4]")
		os.Exit(1)
	}

	m, err := zerfoo.Load(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	// Results are stored in order.
	results := make([]string, len(prompts))
	errs := make([]error, len(prompts))

	// Fan out work across goroutines.
	var wg sync.WaitGroup
	sem := make(chan struct{}, *workers)

	for i, prompt := range prompts {
		wg.Add(1)
		go func(idx int, p string) {
			defer wg.Done()
			sem <- struct{}{}        // Acquire worker slot.
			defer func() { <-sem }() // Release.

			result, err := m.Generate(context.Background(), p,
				zerfoo.WithGenMaxTokens(128),
				zerfoo.WithGenTemperature(0.3),
			)
			if err != nil {
				errs[idx] = err
				return
			}
			results[idx] = result.Text
		}(i, prompt)
	}

	wg.Wait()

	// Print results.
	for i, prompt := range prompts {
		fmt.Printf("--- Prompt %d: %s\n", i+1, prompt)
		if errs[i] != nil {
			fmt.Printf("    Error: %v\n", errs[i])
		} else {
			fmt.Printf("    %s\n\n", results[i])
		}
	}
}
```

## How It Works

1. **Load the model once** -- `zerfoo.Load` loads the GGUF model into memory. The model handle is safe for concurrent use from multiple goroutines.

2. **Fan out with a semaphore** -- A buffered channel acts as a semaphore to limit the number of concurrent inference calls to `--workers` (default 4). Each goroutine acquires a slot before calling `Generate` and releases it when done.

3. **Collect results in order** -- Results and errors are written to pre-allocated slices indexed by prompt position, so the output order matches the input order regardless of which goroutine finishes first.

4. **Configurable generation** -- Each call uses `WithGenMaxTokens` and `WithGenTemperature` to control output length and randomness. You can pass any generation option supported by the API.

## Tuning the Worker Count

- **CPU inference**: Start with `workers = runtime.NumCPU()` and adjust based on memory. Each concurrent generation allocates its own KV cache.
- **GPU inference**: GPU memory is the bottleneck. Start with 1-2 workers and increase until VRAM is saturated.
- **Latency vs. throughput**: More workers increases total throughput but may increase per-request latency due to resource contention.

## Related API Reference

- [Generate API](/docs/api/generate/) -- generation options and result types
- [Inference API](/docs/api/inference/) -- model loading and device selection
