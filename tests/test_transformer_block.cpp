#include "layers/transformer_block.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using mini_inference::layers::MultiHeadAttention;
using mini_inference::layers::RmsNorm;
using mini_inference::layers::SwiGLU;
using mini_inference::layers::TransformerBlock;
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
        if (std::abs(actual - expected) > 1e-4f)
        {
            std::cerr << "FAILED: " << message << " (expected " << expected << ", got " << actual << ")" << std::endl;
            ++failures;
        }
    }

    std::vector<float> identity_weights(std::size_t n)
    {
        std::vector<float> weights(n * n, 0.0f);
        for (std::size_t i = 0; i < n; ++i)
        {
            weights[i * n + i] = 1.0f;
        }
        return weights;
    }

    float silu(float x)
    {
        return x / (1.0f + std::exp(-x));
    }

    TransformerBlock make_block(std::size_t hidden_dim = 2, std::size_t num_heads = 1,
                                 std::size_t intermediate_dim = 1)
    {
        RmsNorm attn_norm(hidden_dim);
        MultiHeadAttention attention(hidden_dim, num_heads, /*causal=*/true, 10000.0f, 8,
                                      identity_weights(hidden_dim), {},
                                      identity_weights(hidden_dim), {},
                                      identity_weights(hidden_dim), {},
                                      identity_weights(hidden_dim), {});
        RmsNorm ffn_norm(hidden_dim);
        SwiGLU ffn(hidden_dim, intermediate_dim,
                   {1.0f, 0.0f}, {},
                   {0.0f, 1.0f}, {},
                   {2.0f, -1.0f}, {});
        return TransformerBlock(std::move(attn_norm), std::move(attention), std::move(ffn_norm), std::move(ffn));
    }

} // namespace

int main()
{
    TransformerBlock block = make_block();

    expect(block.hidden_dim() == 2, "hidden_dim");
    expect(block.num_heads() == 1, "num_heads");
    expect(block.intermediate_dim() == 1, "intermediate_dim");

    // Single-row input: causal self-attention over one row trivially reduces to
    // attn_out == normed1 (softmax over a single score is always 1.0), so the whole
    // pipeline is closed-form and hand-derivable, replicating each sub-layer's own formula.
    const float x0 = 1.0f;
    const float x1 = 2.0f;
    const float eps = 1e-5f;

    const float rms1 = std::sqrt((x0 * x0 + x1 * x1) / 2.0f + eps);
    const float normed1_0 = x0 / rms1;
    const float normed1_1 = x1 / rms1;

    const float residual1_0 = x0 + normed1_0;
    const float residual1_1 = x1 + normed1_1;

    const float rms2 = std::sqrt((residual1_0 * residual1_0 + residual1_1 * residual1_1) / 2.0f + eps);
    const float normed2_0 = residual1_0 / rms2;
    const float normed2_1 = residual1_1 / rms2;

    const float gate = normed2_0;
    const float up = normed2_1;
    const float fused = silu(gate) * up;

    const float expected_out0 = residual1_0 + fused * 2.0f;
    const float expected_out1 = residual1_1 + fused * -1.0f;

    Tensor single_row({1, 2}, {x0, x1});
    Tensor single_output = block.forward(single_row);

    expect(single_output.rank() == 2, "single-row output rank");
    expect(single_output.shape()[0] == 1, "single-row output seq_len");
    expect(single_output.shape()[1] == 2, "single-row output hidden_dim");
    expect_close(single_output.at({0, 0}), expected_out0, "single-row output dim 0");
    expect_close(single_output.at({0, 1}), expected_out1, "single-row output dim 1");

    // Two-row input: shape preservation, and row 0 must match the same closed-form
    // derivation as above regardless of seq_len, since causal masking means row 0
    // never sees row 1 anywhere in the pipeline.
    Tensor two_rows({2, 2}, {x0, x1,
                            0.5f, -0.5f});
    Tensor two_row_output = block.forward(two_rows);

    expect(two_row_output.rank() == 2, "two-row output rank");
    expect(two_row_output.shape()[0] == 2, "two-row output seq_len");
    expect(two_row_output.shape()[1] == 2, "two-row output hidden_dim");
    expect_close(two_row_output.at({0, 0}), expected_out0, "two-row row 0 output dim 0 (causal)");
    expect_close(two_row_output.at({0, 1}), expected_out1, "two-row row 0 output dim 1 (causal)");

    bool threw_rank = false;
    try
    {
        (void)block.forward(Tensor({2}, {1.0f, 2.0f}));
    }
    catch (const std::invalid_argument &)
    {
        threw_rank = true;
    }
    expect(threw_rank, "rank mismatch throws");

    bool threw_dim_mismatch = false;
    try
    {
        (void)block.forward(Tensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    }
    catch (const std::invalid_argument &)
    {
        threw_dim_mismatch = true;
    }
    expect(threw_dim_mismatch, "hidden_dim mismatch throws");

    bool threw_cross_dim_mismatch = false;
    try
    {
        RmsNorm mismatched_attn_norm(4);
        MultiHeadAttention attention(2, 1, /*causal=*/true, 10000.0f, 8,
                                      identity_weights(2), {}, identity_weights(2), {},
                                      identity_weights(2), {}, identity_weights(2), {});
        RmsNorm ffn_norm(2);
        SwiGLU ffn(2, 1, {1.0f, 0.0f}, {}, {0.0f, 1.0f}, {}, {2.0f, -1.0f}, {});
        TransformerBlock bad_block(std::move(mismatched_attn_norm), std::move(attention),
                                    std::move(ffn_norm), std::move(ffn));
        (void)bad_block;
    }
    catch (const std::invalid_argument &)
    {
        threw_cross_dim_mismatch = true;
    }
    expect(threw_cross_dim_mismatch, "cross-sub-layer hidden_dim mismatch throws");

    bool threw_sub_layer_error = false;
    try
    {
        RmsNorm attn_norm(4);
        MultiHeadAttention bad_attention(4, 3); // num_heads does not divide hidden_dim
        RmsNorm ffn_norm(4);
        SwiGLU ffn(4, 1);
        TransformerBlock bad_block(std::move(attn_norm), std::move(bad_attention),
                                    std::move(ffn_norm), std::move(ffn));
        (void)bad_block;
    }
    catch (const std::invalid_argument &)
    {
        threw_sub_layer_error = true;
    }
    expect(threw_sub_layer_error, "sub-layer constructor error propagates uncaught");

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All transformer block tests passed" << std::endl;
    return 0;
}
