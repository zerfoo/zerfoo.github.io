---
title: Numeric Types
weight: 3
bookToc: true
---

# Numeric Types

Zerfoo provides two libraries for reduced-precision floating-point arithmetic: **float16** (IEEE 754 half-precision and BFloat16) and **float8** (FP8 E4M3FN). These are used throughout ztensor for quantized tensor storage and mixed-precision compute.

## At a Glance

| Type | Package | Bits | Format | Range | Precision | Best For |
|------|---------|------|--------|-------|-----------|----------|
| `Float16` | `float16` | 16 | 1 sign + 5 exp + 10 mantissa | ~6.55 x 10^4 | ~3-4 digits | Inference weights, activations |
| `BFloat16` | `float16` | 16 | 1 sign + 8 exp + 7 mantissa | ~3.39 x 10^38 | ~2-3 digits | Training (same range as float32) |
| `Float8` | `float8` | 8 | 1 sign + 4 exp + 3 mantissa (E4M3FN) | ~448 | ~1-2 digits | Quantized inference, memory savings |

## float16

```bash
go get github.com/zerfoo/float16
```

The float16 package provides two types in a single module: `Float16` (IEEE 754 half-precision) and `BFloat16` (Brain Floating Point).

### Float16

Standard IEEE 754 half-precision with 10 bits of mantissa. Good precision for inference weights and activations, but limited range.

```go
import "github.com/zerfoo/float16"

a := float16.FromFloat32(3.14159)
b := float16.FromFloat64(2.71828)

sum := a.Add(b)
product := a.Mul(b)

fmt.Printf("Sum: %f\n", sum.ToFloat32())
fmt.Printf("Product: %f\n", product.ToFloat32())
```

### BFloat16

Same exponent range as float32 (8 exponent bits) with reduced mantissa (7 bits). Preferred for training because it avoids overflow/underflow issues that Float16 suffers from at the edges of the float32 range.

```go
bf := float16.BFloat16FromFloat32(1.5)
f32 := bf.ToFloat32()
```

### Special Values and Classification

```go
f := float16.FromFloat32(3.14)

f.IsInf(0)     // check for infinity
f.IsNaN()      // check for NaN
f.IsFinite()   // check for finite
f.IsNormal()   // check for normalized
f.IsSubnormal() // check for subnormal
```

### Rounding Modes

```go
config := float16.GetConfig()
config.DefaultRoundingMode = float16.RoundNearestEven // default
float16.Configure(config)

// Available: RoundNearestEven, RoundTowardZero,
// RoundTowardPositive, RoundTowardNegative, RoundNearestAway
```

### Vectorized Operations

```go
a := []float16.Float16{...}
b := []float16.Float16{...}

sum := float16.VectorAdd(a, b)
product := float16.VectorMul(a, b)
```

## float8

```bash
go get github.com/zerfoo/float8
```

The float8 package implements FP8 E4M3FN, an 8-bit floating-point format widely used for quantized ML inference. It has no infinity representation (the E4M3FN variant uses that encoding for additional finite values).

```go
import "github.com/zerfoo/float8"

a := float8.FromFloat32(3.14)
b := float8.FromFloat32(2.71)

sum := a.Add(b)
product := a.Mul(b)

fmt.Printf("a + b = %f\n", sum.ToFloat32())
fmt.Printf("a * b = %f\n", product.ToFloat32())
```

### Fast Mode

For performance-critical paths, enable lookup-table-based arithmetic:

```go
float8.EnableFastArithmetic()
float8.EnableFastConversion()
```

This trades memory for speed by using pre-computed tables.

## When to Use Each Type

| Scenario | Recommended Type |
|----------|-----------------|
| Model inference weights | Float16 or BFloat16 |
| Training (mixed precision) | BFloat16 (matches float32 range) |
| Quantized inference (Q8) | Float8 E4M3FN |
| CUDA kernel intermediate values | Float16 |
| Memory-constrained deployment | Float8 |

## Integration with ztensor

These types are first-class citizens in ztensor. Create tensors of any numeric type:

```go
import (
    "github.com/zerfoo/float16"
    "github.com/zerfoo/ztensor/tensor"
    "github.com/zerfoo/ztensor/compute"
    "github.com/zerfoo/ztensor/numeric"
)

// Float16 tensor
a, _ := tensor.New[float16.Float16]([]int{2, 3}, data)
eng := compute.NewCPUEngine[float16.Float16](numeric.Float16Ops{})
```

The compute engine handles dequantization automatically when mixing precision levels.
