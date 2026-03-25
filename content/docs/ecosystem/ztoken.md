---
title: ztoken
weight: 2
bookToc: true
---

# ztoken

BPE tokenizer library for Go with HuggingFace compatibility.

```bash
go get github.com/zerfoo/ztoken
```

## Overview

ztoken provides a Byte-Pair Encoding tokenizer that loads HuggingFace `tokenizer.json` files and extracts tokenizer data from GGUF model files. It handles SentencePiece compatibility (Llama, Gemma), special token management, and text normalization -- with zero external dependencies beyond `golang.org/x/text`.

## Loading a Tokenizer

### From HuggingFace tokenizer.json

```go
import "github.com/zerfoo/ztoken"

tok, err := ztoken.LoadFromJSON("tokenizer.json")
if err != nil {
    panic(err)
}
```

### From a GGUF Model File

The `ztoken/gguf` sub-package extracts tokenizer data directly from GGUF model files, so you don't need a separate JSON file:

```go
import "github.com/zerfoo/ztoken/gguf"

// metadata implements gguf.Metadata interface
tok, err := gguf.ExtractTokenizer(metadata)
if err != nil {
    panic(err)
}
```

### Build Programmatically

```go
vocab := map[string]int{
    "hello": 0, "world": 1, " ": 2,
    "<unk>": 3, "<s>": 4, "</s>": 5, "<pad>": 6,
}
merges := []ztoken.MergePair{
    {Left: "hel", Right: "lo"},
    {Left: "wor", Right: "ld"},
}
special := ztoken.SpecialTokens{BOS: 4, EOS: 5, PAD: 6, UNK: 3}

tok := ztoken.NewBPETokenizer(vocab, merges, special, false)
```

## Encode and Decode

```go
// Encode text to token IDs
ids, _ := tok.Encode("Hello, world!")
fmt.Println(ids)

// Decode token IDs back to text
text, _ := tok.Decode(ids)
fmt.Println(text) // Hello, world!
```

## Special Tokens

```go
special := tok.SpecialTokens()
fmt.Printf("BOS=%d EOS=%d PAD=%d UNK=%d\n",
    special.BOS, special.EOS, special.PAD, special.UNK)

fmt.Println(tok.VocabSize())
```

## SentencePiece Compatibility

Models using SentencePiece tokenization (Llama, Gemma) encode spaces as the U+2581 character. ztoken handles this automatically when loading from GGUF files with `tokenizer.ggml.model = "llama"`, or you can enable it manually:

```go
tok := ztoken.NewBPETokenizer(vocab, merges, special, false)
tok.SetSentencePiece(true)
```

## Supported Models

ztoken is compatible with tokenizers from:

- GPT-2
- Llama 3
- Gemma 3
- Mistral
- Qwen 2
- Phi 3/4
- DeepSeek V3

Any model using BPE with a HuggingFace `tokenizer.json` or GGUF-embedded tokenizer should work.

## Package Structure

| Package | Description |
|---------|-------------|
| `ztoken` | Core tokenizer interface, BPE implementation, HuggingFace JSON loader |
| `ztoken/gguf` | GGUF metadata-based tokenizer extraction |

## Dependencies

ztoken has zero external dependencies beyond the Go standard library and `golang.org/x/text` for Unicode normalization. It is used by [zerfoo](https://github.com/zerfoo/zerfoo) for tokenizing prompts during inference.
