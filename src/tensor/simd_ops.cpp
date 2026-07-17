#include "tensor/simd_ops.h"

#include <cassert>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_INFERENCE_HAVE_NEON 1
#endif

namespace mini_inference::tensor
{

    float dot_product_f32(const float *a, const float *b, std::size_t n)
    {
        assert(a != nullptr && b != nullptr);

        std::size_t i = 0;
        float sum = 0.0f;

#ifdef MINI_INFERENCE_HAVE_NEON
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (; i + 4 <= n; i += 4)
        {
            acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
        }
        sum = vaddvq_f32(acc);
#endif
        for (; i < n; ++i)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    void axpy_f32(float a, const float *x, float *y, std::size_t n)
    {
        assert(x != nullptr && y != nullptr);

        std::size_t i = 0;

#ifdef MINI_INFERENCE_HAVE_NEON
        const float32x4_t va = vdupq_n_f32(a);
        for (; i + 4 <= n; i += 4)
        {
            vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), va, vld1q_f32(x + i)));
        }
#endif
        for (; i < n; ++i)
        {
            y[i] += a * x[i];
        }
    }

    void vector_add_f32(const float *a, const float *b, float *out, std::size_t n)
    {
        assert(a != nullptr && b != nullptr && out != nullptr);

        std::size_t i = 0;

#ifdef MINI_INFERENCE_HAVE_NEON
        for (; i + 4 <= n; i += 4)
        {
            vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        }
#endif
        for (; i < n; ++i)
        {
            out[i] = a[i] + b[i];
        }
    }

    void scale_mul_f32(const float *x, const float *y, float scalar, float *out, std::size_t n)
    {
        assert(x != nullptr && y != nullptr && out != nullptr);

        std::size_t i = 0;

#ifdef MINI_INFERENCE_HAVE_NEON
        const float32x4_t vs = vdupq_n_f32(scalar);
        for (; i + 4 <= n; i += 4)
        {
            vst1q_f32(out + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), vld1q_f32(y + i)), vs));
        }
#endif
        for (; i < n; ++i)
        {
            out[i] = x[i] * y[i] * scalar;
        }
    }

} // namespace mini_inference::tensor
