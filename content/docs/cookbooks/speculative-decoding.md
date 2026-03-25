---
title: Speculative Decoding
weight: 9
bookToc: true
---

# Speculative Decoding

Use a small "draft" model to propose tokens that the large "target" model verifies in parallel. When the draft model guesses correctly (which happens often for common token sequences), multiple tokens are accepted per forward pass, significantly increasing throughput.

## Requirements

- A large target model (e.g. Llama 3 8B)
- A small draft model of the same family (e.g. Llama 3 1B)

## Full Example

```go
// Recipe 09: Speculative Decoding
//
// Use a small "draft" model to propose tokens that the large "target" model
// verifies in parallel. When the draft model guesses correctly (which happens
// often), multiple tokens are accepted per forward pass, significantly
// increasing throughput.
//
// Requirements:
//   - A large target model (e.g. Llama 3 8B)
//   - A small draft model of the same family (e.g. Llama 3 1B)
//
// Usage:
//
//	go run ./docs/cookbook/09-speculative-decoding/ \
//	    --target path/to/llama-8b.gguf \
//	    --draft path/to/llama-1b.gguf
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/serve"
)

func main() {
	targetPath := flag.String("target", "", "path to the large target model (GGUF)")
	draftPath := flag.String("draft", "", "path to the small draft model (GGUF)")
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda"`)
	port := flag.String("port", "8080", "listen port for the OpenAI-compatible API")
	flag.Parse()

	if *targetPath == "" || *draftPath == "" {
		fmt.Fprintln(os.Stderr, "usage: speculative-decoding --target <large.gguf> --draft <small.gguf>")
		os.Exit(1)
	}

	// Load the target (verifier) model.
	target, err := inference.LoadFile(*targetPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "load target: %v\n", err)
		os.Exit(1)
	}
	defer target.Close()

	// Load the draft (proposer) model.
	draft, err := inference.LoadFile(*draftPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "load draft: %v\n", err)
		os.Exit(1)
	}
	defer draft.Close()

	tcfg := target.Config()
	dcfg := draft.Config()
	fmt.Fprintf(os.Stderr, "Target: %s (%d layers)\n", tcfg.Architecture, tcfg.NumLayers)
	fmt.Fprintf(os.Stderr, "Draft:  %s (%d layers)\n", dcfg.Architecture, dcfg.NumLayers)

	// Create a server with speculative decoding enabled.
	// The serve package automatically uses the draft model for speculation
	// when WithDraftModel is provided.
	srv := serve.NewServer(target, serve.WithDraftModel(draft))

	httpServer := &http.Server{
		Addr:    net.JoinHostPort("", *port),
		Handler: srv.Handler(),
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() { errCh <- httpServer.ListenAndServe() }()

	fmt.Fprintf(os.Stderr, "Speculative decoding server on http://localhost:%s\n", *port)

	select {
	case <-ctx.Done():
		fmt.Fprintln(os.Stderr, "\nShutting down...")
		httpServer.Shutdown(context.Background())
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			fmt.Fprintf(os.Stderr, "server: %v\n", err)
			os.Exit(1)
		}
	}
}
```

## How It Works

1. **Load two models** -- The target model is the large, high-quality model you want to serve. The draft model is a smaller, faster model from the same family that shares the same vocabulary.

2. **Enable speculative decoding** -- Pass `serve.WithDraftModel(draft)` when creating the server. The serve package handles the speculation protocol automatically.

3. **Speculation protocol** -- For each generation step:
   - The draft model generates K candidate tokens autoregressively (fast, since it's small).
   - The target model verifies all K candidates in a single batched forward pass.
   - Accepted tokens are emitted immediately. On rejection, generation falls back to the target model's prediction.

4. **Serve over HTTP** -- The server exposes an OpenAI-compatible `/v1/chat/completions` endpoint. Clients see no difference -- speculative decoding is transparent.

## Choosing Draft and Target Models

- The draft and target models should be from the **same family** (e.g., both Llama 3) and share the same tokenizer vocabulary.
- A typical pairing is a 1B draft with an 8B target, or a 3B draft with a 70B target.
- Speedup depends on the acceptance rate: 1.5-3x throughput improvement is typical for well-matched pairs.

## Related API Reference

- [Inference API](/docs/api/inference/) -- `inference.LoadFile` and device options
- [Serve API](/docs/api/serve/) -- server configuration and `WithDraftModel`
