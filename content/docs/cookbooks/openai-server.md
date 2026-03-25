---
title: OpenAI Server
weight: 4
bookToc: true
---

# OpenAI-Compatible Server

Serve a GGUF model behind an OpenAI-compatible HTTP API. Clients that work with the OpenAI API (curl, the Python `openai` library, LangChain, etc.) can connect directly -- just point them at `http://localhost:8080`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion |
| POST | `/v1/completions` | Text completion |
| POST | `/v1/embeddings` | Embeddings |
| GET | `/v1/models` | Model listing |
| GET | `/health` | Health check |

## Usage

```bash
go run ./docs/cookbook/04-openai-server/ --model path/to/model.gguf
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
```

## Full Code

```go
// Recipe 04: OpenAI-Compatible Server
//
// Serve a GGUF model behind an OpenAI-compatible HTTP API. Clients that work
// with the OpenAI API (curl, Python openai library, LangChain, etc.) can
// connect directly -- just point them at http://localhost:8080.
//
// Endpoints:
//   - POST /v1/chat/completions   (chat)
//   - POST /v1/completions        (text completion)
//   - POST /v1/embeddings         (embeddings)
//   - GET  /v1/models             (model listing)
//   - GET  /health                (health check)
//
// Usage:
//
//	go run ./docs/cookbook/04-openai-server/ --model path/to/model.gguf
//	curl http://localhost:8080/v1/chat/completions -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
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
	modelPath := flag.String("model", "", "path to GGUF model file")
	port := flag.String("port", "8080", "listen port")
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda", "cuda:0"`)
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: openai-server --model <model.gguf> [--port 8080] [--device cpu]")
		os.Exit(1)
	}

	// Load the model.
	model, err := inference.LoadFile(*modelPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	cfg := model.Config()
	fmt.Fprintf(os.Stderr, "Loaded %s (%d layers, vocab %d)\n",
		cfg.Architecture, cfg.NumLayers, cfg.VocabSize)

	// Create the OpenAI-compatible server and wire up HTTP.
	srv := serve.NewServer(model)
	httpServer := &http.Server{
		Addr:    net.JoinHostPort("", *port),
		Handler: srv.Handler(),
	}

	// Graceful shutdown on SIGINT / SIGTERM.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() { errCh <- httpServer.ListenAndServe() }()

	fmt.Fprintf(os.Stderr, "Serving on http://localhost:%s\n", *port)

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

1. **Model loading** -- Uses the `inference` package directly (rather than the high-level `zerfoo` package) for full control over device placement via `inference.WithDevice`.
2. **Server creation** -- `serve.NewServer` wraps the loaded model with OpenAI-compatible HTTP handlers for chat, completions, embeddings, and model listing.
3. **Graceful shutdown** -- The server listens for SIGINT/SIGTERM and calls `httpServer.Shutdown` to drain in-flight requests before exiting.
4. **Device selection** -- The `--device` flag supports `"cpu"`, `"cuda"`, or `"cuda:0"` (for multi-GPU systems) to control where inference runs.

## See Also

- [Streaming Chat](/docs/cookbooks/streaming-chat) -- client-side streaming with `ChatStream`
- [Embedding Similarity](/docs/cookbooks/embedding-similarity) -- compute embeddings programmatically
