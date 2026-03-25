---
title: Installation
weight: 1
bookToc: true
---

# Installation

Zerfoo requires **Go 1.25 or later**. [Download Go](https://go.dev/dl/) if you haven't already.

Verify your Go installation:

```bash
go version
# go version go1.25.0 linux/amd64
```

## As a Library

Add Zerfoo to your Go module:

```bash
go get github.com/zerfoo/zerfoo@latest
```

Then import it in your code:

```go
import "github.com/zerfoo/zerfoo"
```

For just tensors and GPU compute, import `github.com/zerfoo/ztensor`. For just tokenization, import `github.com/zerfoo/ztoken`.

## CLI

Install the `zerfoo` command-line tool:

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest
```

This places the `zerfoo` binary in your `$GOPATH/bin` (or `$HOME/go/bin` by default). Make sure this directory is in your `PATH`.

## Build from Source

```bash
git clone https://github.com/zerfoo/zerfoo.git
cd zerfoo
go build -o zerfoo ./cmd/zerfoo
```

Zerfoo builds with **zero CGo by default** (`CGO_ENABLED=0`). GPU acceleration is loaded dynamically at runtime via purego/dlopen, so you do not need CUDA headers, shared libraries, or build tags to compile. A plain `go build` produces a fully static binary.

## Platform Support

Zerfoo compiles on any platform supported by Go 1.25, including **Linux**, **macOS**, and **Windows**.

GPU acceleration is available on:

| Backend | Hardware | Platforms |
|---------|----------|-----------|
| CUDA | NVIDIA GPUs | Linux, Windows |
| ROCm | AMD GPUs | Linux |
| OpenCL | Cross-vendor | Linux, macOS |

For GPU setup instructions, see [GPU Setup](/docs/architecture/gpu-setup/).

## Verify Installation

### Library

Create a test file to confirm the library is importable:

```bash
mkdir zerfoo-test && cd zerfoo-test
go mod init zerfoo-test
go get github.com/zerfoo/zerfoo@latest
```

```go
package main

import (
	"fmt"
	"github.com/zerfoo/zerfoo"
)

func main() {
	fmt.Println("zerfoo imported successfully")
	_ = zerfoo.Load
}
```

```bash
go run main.go
# zerfoo imported successfully
```

### CLI

```bash
zerfoo version
```

This prints the installed version. If the command is not found, ensure `$GOPATH/bin` is in your `PATH`:

```bash
export PATH="$PATH:$(go env GOPATH)/bin"
```

## Next Steps

- [Quickstart](/docs/getting-started/quickstart/) -- pull a model and run your first inference
- [GPU Setup](/docs/architecture/gpu-setup/) -- configure CUDA, ROCm, or OpenCL for hardware-accelerated inference
