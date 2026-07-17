#include "layers/quantized_linear.h"

#include "layers/linear.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using mini_inference::layers::Linear;
using mini_inference::layers::QuantizedLinear;
using mini_inference::tensor::QuantFormat;
using mini_inference::tensor::QuantizedTensor;
using mini_inference::tensor::Tensor;

namespace
{

    int failures = 0;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            std::cerr << "FAILED: " << message << std::endl;
            ++failures;
        }
    }

    void expect_close(float actual, float expected, const std::string &message)
    {
        if (std::abs(actual - expected) > 1e-3f)
        {
            std::cerr << "FAILED: " << message << " (expected " << expected << ", got " << actual << ")" << std::endl;
            ++failures;
        }
    }

    std::vector<std::byte> make_bytes(std::initializer_list<int> values)
    {
        std::vector<std::byte> bytes;
        bytes.reserve(values.size());
        for (int v : values)
        {
            bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(v)));
        }
        return bytes;
    }

    // Two Q8_0 rows of 32 elements: row 0 (out_idx 0) has d=1.0, qs[0]=10 (rest 0); row
    // 1 (out_idx 1) has d=2.0, qs[0]=3, qs[1]=-3 (rest 0). Exactly representable in
    // float, so the quantized and float paths should match exactly, not approximately.
    QuantizedTensor two_row_q8_0_weights()
    {
        std::vector<std::byte> row0 = make_bytes({0x00, 0x3C, 10});
        row0.resize(34, std::byte{0});
        std::vector<std::byte> row1 = make_bytes({0x00, 0x40, 3, 253 /* -3 as int8 */});
        row1.resize(34, std::byte{0});

        std::vector<std::byte> blocks = row0;
        blocks.insert(blocks.end(), row1.begin(), row1.end());
        return QuantizedTensor(QuantFormat::Q8_0, 2, 32, std::move(blocks));
    }

    std::vector<float> equivalent_dense_weights()
    {
        std::vector<float> weights(2 * 32, 0.0f);
        weights[0] = 10.0f;  // row 0, col 0
        weights[32] = 6.0f;  // row 1, col 0
        weights[33] = -6.0f; // row 1, col 1
        return weights;
    }

    // `rows` Q8_0 rows of 32 elements each, every quant = 1 with d = 1.0, so every
    // dequantized row is all-ones. Used to exercise parallel_for's multithreaded path
    // (out_features well above its threading threshold) with an easily-verified result.
    QuantizedTensor many_row_q8_0_weights(std::size_t rows)
    {
        std::vector<std::byte> single_row = make_bytes({0x00, 0x3C}); // d = 1.0 (fp16 0x3C00)
        for (int i = 0; i < 32; ++i)
        {
            single_row.push_back(static_cast<std::byte>(1));
        }

        std::vector<std::byte> blocks;
        blocks.reserve(single_row.size() * rows);
        for (std::size_t r = 0; r < rows; ++r)
        {
            blocks.insert(blocks.end(), single_row.begin(), single_row.end());
        }
        return QuantizedTensor(QuantFormat::Q8_0, rows, 32, std::move(blocks));
    }

} // namespace

int main()
{
    QuantizedLinear quant_linear(two_row_q8_0_weights(), {0.5f, -0.5f});

    expect(quant_linear.in_features() == 32, "in_features");
    expect(quant_linear.out_features() == 2, "out_features");

    Linear dense_linear(32, 2, equivalent_dense_weights(), {0.5f, -0.5f});

    Tensor input({1, 32}, std::vector<float>(32, 1.0f));
    Tensor quant_output = quant_linear.forward(input);
    Tensor dense_output = dense_linear.forward(input);

    expect(quant_output.shape() == dense_output.shape(), "output shapes match");
    expect_close(quant_output.at({0, 0}), dense_output.at({0, 0}), "output 0 matches dense equivalent");
    expect_close(quant_output.at({0, 1}), dense_output.at({0, 1}), "output 1 matches dense equivalent");
    expect_close(quant_output.at({0, 0}), 10.0f + 0.5f, "output 0 hand-derived value");
    expect_close(quant_output.at({0, 1}), 0.0f - 0.5f, "output 1 hand-derived value");

    bool threw_rank = false;
    try
    {
        (void)quant_linear.forward(Tensor({32}, std::vector<float>(32, 1.0f)));
    }
    catch (const std::invalid_argument &)
    {
        threw_rank = true;
    }
    expect(threw_rank, "rank mismatch throws");

    bool threw_bad_bias = false;
    try
    {
        QuantizedLinear bad(two_row_q8_0_weights(), {1.0f, 2.0f, 3.0f});
        (void)bad;
    }
    catch (const std::invalid_argument &)
    {
        threw_bad_bias = true;
    }
    expect(threw_bad_bias, "bias count mismatch throws");

    {
        constexpr std::size_t kOutFeatures = 200;
        QuantizedLinear many_rows(many_row_q8_0_weights(kOutFeatures), {});
        Tensor ones_input({1, 32}, std::vector<float>(32, 1.0f));
        Tensor many_output = many_rows.forward(ones_input);

        expect(many_output.shape()[1] == kOutFeatures, "large out_features output size");
        for (std::size_t out_idx = 0; out_idx < kOutFeatures; ++out_idx)
        {
            expect_close(many_output.at({0, out_idx}), 32.0f,
                         "large out_features row " + std::to_string(out_idx));
        }
    }

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All quantized linear tests passed" << std::endl;
    return 0;
}
