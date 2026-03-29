---
title: ztensor
weight: 1
bookToc: true
---

# ztensor

GPU-accelerated tensor, compute engine, and computation graph library for Go. Current version: **v0.15.0**.

```bash
go get github.com/zerfoo/ztensor
```

## Overview

ztensor is the foundational tensor and compute library in the Zerfoo ecosystem. It provides multi-type tensor storage, a unified compute engine interface across CPU and GPU backends, a computation graph compiler with operator fusion, and GPU memory management -- all without CGo.

If you are building an ML inference engine, need GPU compute from Go, or want a typed tensor library, ztensor is the package to import.

## When to Use ztensor Directly

| Use case | Import |
|----------|--------|
| Tensor math, GPU compute, custom ML operators | `github.com/zerfoo/ztensor` directly |
| Transformer inference, model serving, training | `github.com/zerfoo/zerfoo` (imports ztensor internally) |

Import ztensor directly when you need tensor operations or GPU compute without the full inference/serving stack. If you are running transformer models, use zerfoo -- it builds on ztensor for you.

## Tensor Creation

Tensors are generic over all numeric types via the `tensor.Numeric` constraint:

```go
import "github.com/zerfoo/ztensor/tensor"

// Create a 2x3 float32 tensor
a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

fmt.Println(a.Shape()) // [2, 3]
fmt.Println(a.Data())  // [1 2 3 4 5 6]
```

Supported element types include `float32`, `float64`, `float16.Float16`, `float16.BFloat16`, `float8.Float8`, and all Go integer types.

## Compute Engine

All arithmetic flows through the `compute.Engine[T]` interface. This enables transparent CPU/GPU switching and CUDA graph capture.

### CPU Engine

```go
import (
    "context"

    "github.com/zerfoo/ztensor/compute"
    "github.com/zerfoo/ztensor/numeric"
    "github.com/zerfoo/ztensor/tensor"
)

ctx := context.Background()
eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})

a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
b, _ := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})

c, _ := eng.MatMul(ctx, a, b)
fmt.Println(c.Shape()) // [2, 2]
fmt.Println(c.Data())  // [22 28 49 64]
```

### GPU Engine

GPU libraries are loaded at runtime via purego -- no CGo, no build tags, no linking. If the GPU runtime is not available, the constructor returns an error and you fall back to CPU.

```go
// CUDA (NVIDIA GPUs)
eng, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})

// ROCm (AMD GPUs)
eng, err := compute.NewROCmEngine[float32](numeric.Float32Ops{})

// OpenCL (cross-vendor)
eng, err := compute.NewOpenCLEngine[float32](numeric.Float32Ops{})
```

A common pattern is to try GPU first with a CPU fallback:

```go
eng, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})
if err != nil {
    eng = compute.NewCPUEngine[float32](numeric.Float32Ops{})
}
```

## Type-Safe Generics

Write functions that work across any numeric type:

```go
func dotProduct[T tensor.Numeric](
    eng compute.Engine[T],
    a, b *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
    return eng.MatMul(context.Background(), a, b)
}
```

## Computation Graph

The `graph` package provides a computation graph compiler with operator fusion passes and CUDA graph capture for optimized inference:

| Feature | Description |
|---------|-------------|
| Operator fusion | Combines adjacent operations to reduce kernel launches |
| CUDA graph capture | Records and replays GPU execution for minimal launch overhead |
| Megakernel codegen | Generates fused GPU kernels at compile time |

## Package Reference

| Package | Description |
|---------|-------------|
| `tensor/` | Multi-type tensor storage (CPU, GPU, quantized) |
| `compute/` | Engine interface with CPU, CUDA, ROCm, and OpenCL implementations |
| `graph/` | Computation graph compiler with fusion and CUDA graph capture |
| `numeric/` | Type-safe `Arithmetic[T]` interface for all numeric types |
| `device/` | Device abstraction and memory allocators |
| `internal/cuda/` | Zero-CGo CUDA runtime bindings via purego, 25+ custom kernels |
| `internal/xblas/` | ARM NEON and x86 AVX2 SIMD assembly |
| `internal/gpuapi/` | GPU Runtime Abstraction Layer (CUDA/ROCm/OpenCL) |
| `internal/codegen/` | Megakernel code generator |

## What's New in v0.15.0

### MmapStorage.SliceElements

`MmapStorage.SliceElements` provides zero-copy slicing of mmap'd tensor elements. It returns a view into the memory-mapped region without copying data, making expert weight extraction in mixture-of-experts models efficient:

```go
// Extract expert weights directly from the mmap'd file — no allocation
expertWeights, err := mmapStorage.SliceElements(expertOffset, expertSize)
```

This replaces the previous pattern of copying expert weights into a new tensor before each forward pass.

### Streaming GEMM for mmap'd Tensors

`internal/xblas` now includes a streaming GEMM path for mmap'd weight tensors. Instead of paging in the entire weight matrix before computation, the kernel tiles over the mmap region in cache-sized chunks, keeping memory bandwidth proportional to the active tile rather than the full matrix.

This enables over-RAM CPU inference: a model whose weights exceed physical RAM can run without GPU, with the OS paging tensor data from NVMe on demand. Combined with `MmapStorage.SliceElements`, a 229B MoE model runs on a 128 GB machine with no configuration flags.

## Dependencies

ztensor depends on [float16]({{< relref "numeric-types" >}}) and [float8]({{< relref "numeric-types" >}}) for half-precision and FP8 arithmetic.
