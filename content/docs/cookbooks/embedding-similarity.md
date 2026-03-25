---
title: Embedding Similarity
weight: 3
bookToc: true
---

# Embedding and Cosine Similarity

Compute text embeddings and rank a corpus of documents by relevance to a query using cosine similarity. This is the retrieval building block for semantic search and RAG systems.

## Usage

```bash
go run ./docs/cookbook/03-embedding-similarity/ --model path/to/model.gguf
go run ./docs/cookbook/03-embedding-similarity/ --model path/to/model.gguf --query "memory management"
```

## Full Code

```go
// Recipe 03: Embedding and Cosine Similarity
//
// Compute text embeddings and rank a corpus of documents by relevance to a
// query using cosine similarity. This is the retrieval building block for
// semantic search and RAG systems.
//
// Usage:
//
//	go run ./docs/cookbook/03-embedding-similarity/ --model path/to/model.gguf
//	go run ./docs/cookbook/03-embedding-similarity/ --model path/to/model.gguf --query "memory management"
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"

	"github.com/zerfoo/zerfoo"
)

// corpus is a small document set for demonstration.
var corpus = []string{
	"Go's garbage collector is a concurrent, tri-color, mark-sweep collector.",
	"Goroutines are multiplexed onto OS threads by the Go runtime scheduler.",
	"The sync.Mutex type provides mutual exclusion for shared state.",
	"Go modules use go.mod and go.sum to manage versioned dependencies.",
	"Channels are the primary mechanism for goroutine communication.",
	"The context package carries deadlines and cancellation signals.",
	"Go interfaces are satisfied implicitly without an implements keyword.",
	"The testing package supports automated unit and benchmark tests.",
}

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file or HuggingFace model ID")
	query := flag.String("query", "How does Go handle concurrency?", "search query")
	topN := flag.Int("top", 3, "number of results to display")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: embedding-similarity --model <path> [--query <text>]")
		os.Exit(1)
	}

	m, err := zerfoo.Load(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	// Embed all documents in the corpus.
	corpusEmbeds, err := m.Embed(corpus)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embed corpus: %v\n", err)
		os.Exit(1)
	}

	// Embed the query.
	queryEmbeds, err := m.Embed([]string{*query})
	if err != nil {
		fmt.Fprintf(os.Stderr, "embed query: %v\n", err)
		os.Exit(1)
	}
	qe := queryEmbeds[0]

	// Rank documents by cosine similarity.
	type result struct {
		doc   string
		score float32
	}
	results := make([]result, len(corpus))
	for i, emb := range corpusEmbeds {
		results[i] = result{corpus[i], qe.CosineSimilarity(emb)}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })

	fmt.Printf("Query: %q\n\n", *query)
	n := *topN
	if n > len(results) {
		n = len(results)
	}
	for i := 0; i < n; i++ {
		fmt.Printf("  %d. [%.4f] %s\n", i+1, results[i].score, results[i].doc)
	}
}
```

## How It Works

1. **Embedding** -- `m.Embed` takes a slice of strings and returns a slice of embedding vectors. Each vector captures the semantic meaning of the input text.
2. **Cosine similarity** -- The `CosineSimilarity` method computes the cosine of the angle between two embedding vectors. Values closer to 1.0 indicate higher semantic similarity.
3. **Ranking** -- Documents are sorted by their similarity score to the query, and the top N results are displayed.

This pattern is the foundation for retrieval-augmented generation (RAG): embed your document corpus, embed the user query, retrieve the most relevant documents, and pass them as context to the generation step.

## See Also

- [Basic Text Generation](/docs/cookbooks/basic-text-generation) -- generate text completions
- [OpenAI Server](/docs/cookbooks/openai-server) -- the `/v1/embeddings` endpoint serves embeddings over HTTP
