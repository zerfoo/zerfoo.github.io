---
title: Vision / Multimodal
weight: 12
bookToc: true
---

# Vision / Multimodal Inference

Analyze images using a vision-capable GGUF model. The image is passed alongside a text prompt using the `inference.Message` API, the same format used by the OpenAI-compatible `/v1/chat/completions` endpoint.

## Requirements

- A vision-capable GGUF model (e.g. LLaVA, Gemma 3 with vision encoder)

## Full Example

```go
// Recipe 12: Vision / Multimodal Inference
//
// Analyze images using a vision-capable GGUF model. The image is passed
// alongside a text prompt using the inference.Message API, the same format
// used by the OpenAI-compatible /v1/chat/completions endpoint.
//
// Requirements:
//   - A vision-capable GGUF model (e.g. LLaVA, Gemma 3 with vision encoder)
//
// Usage:
//
//	go run ./docs/cookbook/12-vision-multimodal/ --model path/to/vision-model.gguf --image photo.jpg
//	go run ./docs/cookbook/12-vision-multimodal/ --model path/to/vision-model.gguf --image photo.jpg --prompt "Count the objects"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	modelPath := flag.String("model", "", "path to a vision-capable GGUF model")
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda"`)
	imagePath := flag.String("image", "", "path to an image file (JPEG or PNG)")
	prompt := flag.String("prompt", "Describe this image in detail.", "question about the image")
	maxTokens := flag.Int("max-tokens", 512, "maximum tokens to generate")
	flag.Parse()

	if *modelPath == "" || *imagePath == "" {
		fmt.Fprintln(os.Stderr, "usage: vision-multimodal --model <model.gguf> --image <image.jpg>")
		os.Exit(1)
	}

	// Read the image file into memory.
	imageData, err := os.ReadFile(*imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read image: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Image: %s (%d bytes)\n", *imagePath, len(imageData))

	// Load the vision-capable model.
	model, err := inference.LoadFile(*modelPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	cfg := model.Config()
	fmt.Fprintf(os.Stderr, "Model: %s (%d layers)\n", cfg.Architecture, cfg.NumLayers)

	// Build a chat message with the image embedded.
	// The Images field carries raw image bytes, matching the OpenAI vision API.
	messages := []inference.Message{
		{
			Role:    "user",
			Content: *prompt,
			Images:  [][]byte{imageData},
		},
	}

	resp, err := model.Chat(context.Background(), messages,
		inference.WithMaxTokens(*maxTokens),
		inference.WithTemperature(0.5),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(resp.Content)
}
```

## How It Works

1. **Read the image** -- The image file (JPEG or PNG) is read into a byte slice with `os.ReadFile`. No preprocessing is needed -- the model handles decoding and resizing internally.

2. **Load a vision model** -- Use `inference.LoadFile` with a vision-capable GGUF model. Models like LLaVA and Gemma 3 with vision encoders include both a vision tower (image encoder) and a language model in a single GGUF file.

3. **Build a multimodal message** -- The `inference.Message` struct accepts both text (`Content`) and images (`Images` as raw bytes). This mirrors the OpenAI chat completions API where images are passed inline.

4. **Generate** -- `model.Chat` processes the image through the vision encoder, projects the image embeddings into the language model's space, and generates a text response conditioned on both the image and the text prompt.

## Supported Vision Models

| Model | Architecture | Notes |
|-------|-------------|-------|
| LLaVA | LLaVA | CLIP vision encoder + Llama language model |
| Gemma 3 (with vision) | Gemma 3 | SigLIP vision encoder + Gemma language model |

## Use Cases

- **Image captioning**: "Describe this image in detail."
- **Visual question answering**: "How many people are in this photo?"
- **Document understanding**: "Extract the text from this receipt."
- **Object detection**: "List all objects visible in this image."

## Related API Reference

- [Inference API](/docs/api/inference/) -- `inference.LoadFile`, `inference.Message`, and `model.Chat`
- [Serve API](/docs/api/serve/) -- the `/v1/chat/completions` endpoint accepts images via the OpenAI vision API format
