#pragma once

#include <cstddef>

namespace mini_inference::tensor
{

    // Small set of SIMD-accelerated (ARM NEON, with scalar fallback) primitives that
    // the rest of the forward pass (Linear, QuantizedLinear, attention, RmsNorm,
    // residual add) is built from. Operate on raw contiguous float buffers rather than
    // Tensor so callers control exactly which rows/slices are combined.

    // out = sum(a[i]*b[i]) for i in 0..n.
    float dot_product_f32(const float *a, const float *b, std::size_t n);

    // y[i] += a * x[i] for i in 0..n (saxpy).
    void axpy_f32(float a, const float *x, float *y, std::size_t n);

    // out[i] = a[i] + b[i] for i in 0..n.
    void vector_add_f32(const float *a, const float *b, float *out, std::size_t n);

    // out[i] = x[i] * y[i] * scalar for i in 0..n.
    void scale_mul_f32(const float *x, const float *y, float scalar, float *out, std::size_t n);

} // namespace mini_inference::tensor
