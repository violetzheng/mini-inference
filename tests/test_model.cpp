#include "model/model.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using mini_inference::layers::Embedding;
using mini_inference::layers::Linear;
using mini_inference::layers::MultiHeadAttention;
using mini_inference::layers::RmsNorm;
using mini_inference::layers::SwiGLU;
using mini_inference::layers::TransformerBlock;
using mini_inference::model::Model;
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

    // Identity-attention + sparse-SwiGLU block, identical to test_transformer_block.cpp's
    // make_block(), so its closed-form single-row derivation can be reused verbatim here.
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

    Embedding make_embedding(std::size_t vocab_size = 3, std::size_t hidden_dim = 2)
    {
        // Row 0 == {1.0f, 2.0f}, the exact input hand-derived in test_transformer_block.cpp.
        return Embedding(vocab_size, hidden_dim,
                          {1.0f, 2.0f,
                           0.0f, 0.0f,
                           -1.0f, -1.0f});
    }

    Linear make_lm_head(std::size_t hidden_dim = 2, std::size_t vocab_size = 3)
    {
        return Linear(hidden_dim, vocab_size, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, {});
    }

} // namespace

int main()
{
    // num_layers = 1: fully hand-derived end-to-end test, chaining embedding lookup ->
    // the same closed-form TransformerBlock derivation from test_transformer_block.cpp ->
    // final RmsNorm -> lm_head projection.
    std::vector<TransformerBlock> single_block;
    single_block.push_back(make_block());
    Model model(make_embedding(), std::move(single_block), RmsNorm(2), make_lm_head());

    expect(model.vocab_size() == 3, "vocab_size");
    expect(model.hidden_dim() == 2, "hidden_dim");
    expect(model.num_layers() == 1, "num_layers");

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

    const float fused = silu(normed2_0) * normed2_1;

    const float block_out0 = residual1_0 + fused * 2.0f;
    const float block_out1 = residual1_1 + fused * -1.0f;

    const float final_rms = std::sqrt((block_out0 * block_out0 + block_out1 * block_out1) / 2.0f + eps);
    const float final0 = block_out0 / final_rms;
    const float final1 = block_out1 / final_rms;

    const float expected_logit0 = final0;
    const float expected_logit1 = final1;
    const float expected_logit2 = final0 + final1;

    Tensor output = model.forward({0});

    expect(output.rank() == 2, "single-layer output rank");
    expect(output.shape()[0] == 1, "single-layer output seq_len");
    expect(output.shape()[1] == 3, "single-layer output vocab_size");
    expect_close(output.at({0, 0}), expected_logit0, "single-layer logit 0");
    expect_close(output.at({0, 1}), expected_logit1, "single-layer logit 1");
    expect_close(output.at({0, 2}), expected_logit2, "single-layer logit 2");

    // num_layers = 2: shape check plus a sanity check that stacking actually iterates
    // (output must differ from the num_layers = 1 result for the same input).
    std::vector<TransformerBlock> two_blocks;
    two_blocks.push_back(make_block());
    two_blocks.push_back(make_block());
    Model stacked_model(make_embedding(), std::move(two_blocks), RmsNorm(2), make_lm_head());

    expect(stacked_model.num_layers() == 2, "stacked num_layers");

    Tensor stacked_output = stacked_model.forward({0});
    expect(stacked_output.rank() == 2, "stacked output rank");
    expect(stacked_output.shape()[0] == 1, "stacked output seq_len");
    expect(stacked_output.shape()[1] == 3, "stacked output vocab_size");
    expect(std::abs(stacked_output.at({0, 0}) - output.at({0, 0})) > 1e-4f,
           "stacking two blocks changes the output vs. one block");

    bool threw_empty_tokens = false;
    try
    {
        (void)model.forward({});
    }
    catch (const std::invalid_argument &)
    {
        threw_empty_tokens = true;
    }
    expect(threw_empty_tokens, "empty token_ids throws");

    bool threw_out_of_range_token = false;
    try
    {
        (void)model.forward({100});
    }
    catch (const std::invalid_argument &)
    {
        threw_out_of_range_token = true;
    }
    expect(threw_out_of_range_token, "out-of-range token id propagates from embedding");

    bool threw_zero_blocks = false;
    try
    {
        Model bad_model(make_embedding(), std::vector<TransformerBlock>{}, RmsNorm(2), make_lm_head());
        (void)bad_model;
    }
    catch (const std::invalid_argument &)
    {
        threw_zero_blocks = true;
    }
    expect(threw_zero_blocks, "zero transformer blocks throws");

    bool threw_cross_dim_mismatch = false;
    try
    {
        std::vector<TransformerBlock> blocks;
        blocks.push_back(make_block());
        Model bad_model(make_embedding(), std::move(blocks), RmsNorm(4), make_lm_head());
        (void)bad_model;
    }
    catch (const std::invalid_argument &)
    {
        threw_cross_dim_mismatch = true;
    }
    expect(threw_cross_dim_mismatch, "final_norm hidden_dim mismatch throws");

    bool threw_vocab_mismatch = false;
    try
    {
        std::vector<TransformerBlock> blocks;
        blocks.push_back(make_block());
        Model bad_model(make_embedding(3, 2), std::move(blocks), RmsNorm(2), make_lm_head(2, 4));
        (void)bad_model;
    }
    catch (const std::invalid_argument &)
    {
        threw_vocab_mismatch = true;
    }
    expect(threw_vocab_mismatch, "lm_head out_features / vocab_size mismatch throws");

    bool threw_sub_layer_error = false;
    try
    {
        std::vector<TransformerBlock> blocks;
        blocks.push_back(make_block(4, 3)); // num_heads does not divide hidden_dim
        Model bad_model(make_embedding(3, 4), std::move(blocks), RmsNorm(4), make_lm_head(4, 3));
        (void)bad_model;
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

    std::cout << "All model tests passed" << std::endl;
    return 0;
}
