#include "model/model.h"

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

    std::vector<float> identity_weights(std::size_t n)
    {
        std::vector<float> weights(n * n, 0.0f);
        for (std::size_t i = 0; i < n; ++i)
        {
            weights[i * n + i] = 1.0f;
        }
        return weights;
    }

    // Same identity-attention + sparse-SwiGLU block as test_model.cpp's make_block(),
    // so single-token behavior is hand-derivable and matches that file's derivation.
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
        // Row 0 == {1.0f, 2.0f}, the exact input hand-derived in test_model.cpp, so that
        // model.generate({0}, ...) reuses the same known first-step logits.
        return Embedding(vocab_size, hidden_dim,
                          {1.0f, 2.0f,
                           0.0f, 0.0f,
                           -1.0f, -1.0f});
    }

    Linear make_lm_head(std::size_t hidden_dim = 2, std::size_t vocab_size = 3)
    {
        return Linear(hidden_dim, vocab_size, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, {});
    }

    Model make_model()
    {
        std::vector<TransformerBlock> blocks;
        blocks.push_back(make_block());
        return Model(make_embedding(), std::move(blocks), RmsNorm(2), make_lm_head());
    }

    // Duplicated on purpose, mirroring Model::generate's internal argmax helper, so this
    // test can independently cross-check generate()'s loop/append logic against forward().
    std::size_t argmax_last_row(const Tensor &logits)
    {
        const std::size_t last_row = logits.shape()[0] - 1;
        const std::size_t vocab_size = logits.shape()[1];

        std::size_t best_index = 0;
        float best_value = logits.at({last_row, 0});
        for (std::size_t v = 1; v < vocab_size; ++v)
        {
            const float value = logits.at({last_row, v});
            if (value > best_value)
            {
                best_value = value;
                best_index = v;
            }
        }
        return best_index;
    }

} // namespace

int main()
{
    Model model = make_model();

    // Step-1 logits for token 0 are identical to test_model.cpp's hand-derived
    // expected_logit0/1/2. Both final0 and final1 end up positive there (block_out0 and
    // block_out1 are each a positive residual plus/minus a much smaller fused term), so
    // logit2 = logit0 + logit1 is necessarily the largest: argmax is always index 2,
    // without depending on exact floating-point values.
    const std::size_t expected_first_token = 2;

    std::vector<std::size_t> single_step = model.generate({0}, 1);
    expect(single_step.size() == 2, "single-step generate length");
    expect(single_step[0] == 0, "single-step generate keeps prompt token");
    expect(single_step[1] == expected_first_token, "single-step generate picks argmax token");

    // Multi-step: independently loop forward() + argmax outside of generate() to build
    // an expected sequence, and check generate() matches it. This validates the new
    // loop/append orchestration without re-deriving multi-token causal attention by hand.
    std::vector<std::size_t> expected_multi_step = {0};
    for (int step = 0; step < 3; ++step)
    {
        Tensor logits = model.forward(expected_multi_step);
        expected_multi_step.push_back(argmax_last_row(logits));
    }

    std::vector<std::size_t> multi_step = model.generate({0}, 3);
    expect(multi_step == expected_multi_step, "multi-step generate matches independent forward+argmax loop");

    std::vector<std::size_t> zero_new_tokens = model.generate({0}, 0);
    expect(zero_new_tokens.size() == 1 && zero_new_tokens[0] == 0,
           "max_new_tokens = 0 returns prompt unchanged");

    // eos_token_id equal to the known first argmax token should stop generation right
    // after it is produced, well before the max_new_tokens budget is exhausted.
    std::vector<std::size_t> stopped_early = model.generate({0}, 5, expected_first_token);
    expect(stopped_early.size() == 2, "eos_token_id stops generation early");
    expect(stopped_early[1] == expected_first_token, "eos_token_id is the last generated token");

    bool threw_empty_prompt = false;
    try
    {
        (void)model.generate({}, 3);
    }
    catch (const std::invalid_argument &)
    {
        threw_empty_prompt = true;
    }
    expect(threw_empty_prompt, "empty prompt throws");

    bool threw_out_of_range_token = false;
    try
    {
        (void)model.generate({100}, 1);
    }
    catch (const std::invalid_argument &)
    {
        threw_out_of_range_token = true;
    }
    expect(threw_out_of_range_token, "out-of-range prompt token id propagates from forward/embedding");

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All generate tests passed" << std::endl;
    return 0;
}
