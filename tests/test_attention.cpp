#include "layers/attention.h"
#include "layers/kv_cache.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using mini_inference::layers::KvCache;
using mini_inference::layers::Linear;
using mini_inference::layers::MultiHeadAttention;
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

    // GQA sanity check: hidden_dim=4, num_heads=2, num_kv_heads=1 (head_dim=2, kv_dim=2).
    // K selects input dims [0,1], V selects input dims [2,3], Q/O are identity. With a
    // single-row input, causal softmax over one position is always weight 1.0 regardless
    // of the score, so both query heads (which share the single KV head under GQA) must
    // each output exactly V's value verbatim - this confirms head 1 doesn't silently read
    // from head 1 of a hidden_dim_-wide K/V (there is no such head under GQA).
    void test_grouped_query_attention()
    {
        const std::size_t hidden_dim = 4;
        const std::size_t num_heads = 2;
        const std::size_t num_kv_heads = 1;

        const std::vector<float> k_weights = {1, 0, 0, 0,
                                               0, 1, 0, 0};
        const std::vector<float> v_weights = {0, 0, 1, 0,
                                               0, 0, 0, 1};

        MultiHeadAttention attn(hidden_dim, num_heads, /*causal=*/true, 10000.0f, 8,
                                 identity_weights(hidden_dim), {},
                                 k_weights, {},
                                 v_weights, {},
                                 identity_weights(hidden_dim), {},
                                 num_kv_heads);

        expect(attn.num_heads() == 2, "gqa num_heads");
        expect(attn.num_kv_heads() == 1, "gqa num_kv_heads");
        expect(attn.head_dim() == 2, "gqa head_dim");
        expect(attn.kv_dim() == 2, "gqa kv_dim");

        Tensor input({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
        Tensor output = attn.forward(input);

        expect_close(output.at({0, 0}), 3.0f, "gqa head 0 shares kv head 0 (dim 0)");
        expect_close(output.at({0, 1}), 4.0f, "gqa head 0 shares kv head 0 (dim 1)");
        expect_close(output.at({0, 2}), 3.0f, "gqa head 1 shares kv head 0 (dim 0)");
        expect_close(output.at({0, 3}), 4.0f, "gqa head 1 shares kv head 0 (dim 1)");

        // Same expectation through the KV-cache path.
        KvCache cache(4, attn.kv_dim());
        Tensor cached_output = attn.forward(input, cache);
        expect_close(cached_output.at({0, 0}), 3.0f, "gqa cache path head 0 (dim 0)");
        expect_close(cached_output.at({0, 1}), 4.0f, "gqa cache path head 0 (dim 1)");
        expect_close(cached_output.at({0, 2}), 3.0f, "gqa cache path head 1 (dim 0)");
        expect_close(cached_output.at({0, 3}), 4.0f, "gqa cache path head 1 (dim 1)");

        bool threw_indivisible = false;
        try
        {
            // hidden_dim=4, num_heads=4 divides evenly (head_dim=1), but num_heads=4 is
            // not a multiple of num_kv_heads=3, so this must fail on the GQA check
            // specifically, not on the unrelated hidden_dim/num_heads divisibility check.
            MultiHeadAttention bad(4, 4, true, 10000.0f, 8, {}, {}, {}, {}, {}, {}, {}, {}, /*num_kv_heads=*/3);
            (void)bad;
        }
        catch (const std::invalid_argument &)
        {
            threw_indivisible = true;
        }
        expect(threw_indivisible, "num_heads not divisible by num_kv_heads throws");
    }

} // namespace

int main()
{
    // With identity Q/K/V/O weights and no bias, attention reduces to a pure
    // function of the rotated queries/keys and the (unrotated) values, so the
    // expected values can be hand-derived from the same cos/sin formulas RoPE uses.
    const std::size_t hidden_dim = 2;
    const std::size_t num_heads = 1;
    MultiHeadAttention attn(hidden_dim, num_heads, /*causal=*/true, 10000.0f, 8,
                             identity_weights(hidden_dim), {},
                             identity_weights(hidden_dim), {},
                             identity_weights(hidden_dim), {},
                             identity_weights(hidden_dim), {});

    expect(attn.hidden_dim() == 2, "hidden_dim");
    expect(attn.num_heads() == 1, "num_heads");
    expect(attn.head_dim() == 2, "head_dim");
    expect(attn.causal(), "causal");

    Tensor input({2, 2}, {1.0f, 0.0f,
                          0.0f, 1.0f});
    Tensor output = attn.forward(input);

    expect(output.rank() == 2, "output rank");
    expect(output.shape()[0] == 2, "output seq_len");
    expect(output.shape()[1] == 2, "output hidden_dim");

    // Row 0 causally attends only to itself, so it must equal V row 0 exactly,
    // regardless of RoPE (rotation angle at position 0 is zero anyway).
    expect_close(output.at({0, 0}), 1.0f, "row 0 attends only to itself (dim 0)");
    expect_close(output.at({0, 1}), 0.0f, "row 0 attends only to itself (dim 1)");

    // Row 1: Q1/K1 are row 1 rotated by angle = 1 * inv_freq(0) = 1 radian.
    const float angle = 1.0f;
    const float cos_a = std::cos(angle);
    const float sin_a = std::sin(angle);
    // Row 1 input is [0, 1] -> rotated: (x_even', x_odd') = (-sin, cos).
    const float q1_even = -sin_a;
    const float q1_odd = cos_a;

    const float scale = 1.0f / std::sqrt(static_cast<float>(hidden_dim));
    // K0 is row 0 at position 0 (identity rotation): [1, 0].
    const float score10 = (q1_even * 1.0f + q1_odd * 0.0f) * scale;
    // K1 == Q1 here (identical projections and rotation), dot(Q1, Q1) = |row1|^2 = 1.
    const float score11 = (q1_even * q1_even + q1_odd * q1_odd) * scale;

    const float max_score = std::max(score10, score11);
    const float exp10 = std::exp(score10 - max_score);
    const float exp11 = std::exp(score11 - max_score);
    const float sum_exp = exp10 + exp11;
    const float p10 = exp10 / sum_exp;
    const float p11 = exp11 / sum_exp;

    // out_row1 = p10 * V0 + p11 * V1 = p10 * [1, 0] + p11 * [0, 1].
    expect_close(output.at({1, 0}), p10, "row 1 attention-weighted output (dim 0)");
    expect_close(output.at({1, 1}), p11, "row 1 attention-weighted output (dim 1)");

    // Non-causal attention over a single row must equal causal (nothing to mask).
    MultiHeadAttention non_causal(hidden_dim, num_heads, /*causal=*/false, 10000.0f, 8,
                                   identity_weights(hidden_dim), {},
                                   identity_weights(hidden_dim), {},
                                   identity_weights(hidden_dim), {},
                                   identity_weights(hidden_dim), {});
    Tensor single_row({1, 2}, {1.0f, 0.0f});
    Tensor causal_single = attn.forward(single_row);
    Tensor non_causal_single = non_causal.forward(single_row);
    expect_close(causal_single.at({0, 0}), non_causal_single.at({0, 0}), "causal == non-causal for seq_len 1 (dim 0)");
    expect_close(causal_single.at({0, 1}), non_causal_single.at({0, 1}), "causal == non-causal for seq_len 1 (dim 1)");

    bool threw_rank = false;
    try
    {
        (void)attn.forward(Tensor({2}, {1.0f, 2.0f}));
    }
    catch (const std::invalid_argument &)
    {
        threw_rank = true;
    }
    expect(threw_rank, "rank mismatch throws");

    bool threw_dim_mismatch = false;
    try
    {
        (void)attn.forward(Tensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    }
    catch (const std::invalid_argument &)
    {
        threw_dim_mismatch = true;
    }
    expect(threw_dim_mismatch, "hidden_dim mismatch throws");

    bool threw_bad_heads = false;
    try
    {
        MultiHeadAttention bad_attn(4, 3);
        (void)bad_attn;
    }
    catch (const std::invalid_argument &)
    {
        threw_bad_heads = true;
    }
    expect(threw_bad_heads, "hidden_dim not divisible by num_heads throws");

    bool threw_zero_heads = false;
    try
    {
        MultiHeadAttention zero_heads(4, 0);
        (void)zero_heads;
    }
    catch (const std::invalid_argument &)
    {
        threw_zero_heads = true;
    }
    expect(threw_zero_heads, "zero num_heads throws");

    test_grouped_query_attention();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All multi-head attention tests passed" << std::endl;
    return 0;
}
