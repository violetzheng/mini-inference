#include "tensor/simd_ops.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using mini_inference::tensor::axpy_f32;
using mini_inference::tensor::dot_product_f32;
using mini_inference::tensor::scale_mul_f32;
using mini_inference::tensor::vector_add_f32;

namespace
{

    int failures = 0;

    void expect_close(float actual, float expected, const std::string &message)
    {
        if (std::abs(actual - expected) > 1e-3f)
        {
            std::cerr << "FAILED: " << message << " (expected " << expected << ", got " << actual << ")" << std::endl;
            ++failures;
        }
    }

    // 1, 2, 3, ... n as floats, used to build predictable inputs for every test below.
    std::vector<float> iota(std::size_t n, float start = 1.0f)
    {
        std::vector<float> values(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            values[i] = start + static_cast<float>(i);
        }
        return values;
    }

    void test_dot_product_multiple_of_four()
    {
        const std::vector<float> a = iota(8);              // 1..8
        const std::vector<float> b = iota(8, /*start=*/2);  // 2..9
        // sum(a[i]*b[i]) = 1*2 + 2*3 + ... + 8*9 = 2+6+12+20+30+42+56+72 = 240
        expect_close(dot_product_f32(a.data(), b.data(), a.size()), 240.0f, "dot_product_f32 n=8 (multiple of 4)");
    }

    void test_dot_product_remainder_tail()
    {
        const std::vector<float> a = iota(13);
        const std::vector<float> b(13, 1.0f);
        // sum(1..13) = 91
        expect_close(dot_product_f32(a.data(), b.data(), a.size()), 91.0f, "dot_product_f32 n=13 (remainder tail)");
    }

    void test_axpy_accumulates_into_existing_y()
    {
        const std::vector<float> x = iota(9); // 1..9
        std::vector<float> y(9, 10.0f);       // pre-populated, non-zero
        axpy_f32(2.0f, x.data(), y.data(), y.size());
        // y[i] = 10 + 2*x[i]
        for (std::size_t i = 0; i < y.size(); ++i)
        {
            expect_close(y[i], 10.0f + 2.0f * x[i], "axpy_f32 element " + std::to_string(i));
        }
    }

    void test_vector_add()
    {
        const std::vector<float> a = iota(11);
        const std::vector<float> b = iota(11, /*start=*/100.0f);
        std::vector<float> out(11, 0.0f);
        vector_add_f32(a.data(), b.data(), out.data(), out.size());
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            expect_close(out[i], a[i] + b[i], "vector_add_f32 element " + std::to_string(i));
        }
    }

    void test_scale_mul()
    {
        const std::vector<float> x = iota(15);
        const std::vector<float> y = iota(15, /*start=*/0.5f);
        const float scalar = 3.0f;
        std::vector<float> out(15, 0.0f);
        scale_mul_f32(x.data(), y.data(), scalar, out.data(), out.size());
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            expect_close(out[i], x[i] * y[i] * scalar, "scale_mul_f32 element " + std::to_string(i));
        }
    }

} // namespace

int main()
{
    test_dot_product_multiple_of_four();
    test_dot_product_remainder_tail();
    test_axpy_accumulates_into_existing_y();
    test_vector_add();
    test_scale_mul();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All simd_ops tests passed" << std::endl;
    return 0;
}
