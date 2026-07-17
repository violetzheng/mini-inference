#include "layers/kv_cache.h"

#include "layers/attention.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using mini_inference::layers::KvCache;
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

    void test_kv_cache_basics()
    {
        KvCache cache(4, 2);
        expect(cache.length() == 0, "fresh cache length is zero");
        expect(cache.max_seq_len() == 4, "max_seq_len accessor");
        expect(cache.hidden_dim() == 2, "hidden_dim accessor");

        Tensor k0({1, 2}, {1.0f, 2.0f});
        Tensor v0({1, 2}, {3.0f, 4.0f});
        cache.append(k0, v0);
        expect(cache.length() == 1, "length after first append");

        Tensor k1({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
        Tensor v1({2, 2}, {9.0f, 10.0f, 11.0f, 12.0f});
        cache.append(k1, v1);
        expect(cache.length() == 3, "length after second append");

        const Tensor keys = cache.keys();
        const Tensor values = cache.values();
        expect(keys.shape()[0] == 3 && keys.shape()[1] == 2, "keys shape");
        expect(values.shape()[0] == 3 && values.shape()[1] == 2, "values shape");
        expect_close(keys.at({0, 0}), 1.0f, "keys row 0 col 0");
        expect_close(keys.at({0, 1}), 2.0f, "keys row 0 col 1");
        expect_close(keys.at({2, 0}), 7.0f, "keys row 2 col 0");
        expect_close(keys.at({2, 1}), 8.0f, "keys row 2 col 1");
        expect_close(values.at({1, 0}), 9.0f, "values row 1 col 0");
        expect_close(values.at({1, 1}), 10.0f, "values row 1 col 1");

        bool threw_overflow = false;
        try
        {
            Tensor k2({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});
            Tensor v2({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});
            cache.append(k2, v2);
        }
        catch (const std::out_of_range &)
        {
            threw_overflow = true;
        }
        expect(threw_overflow, "append past max_seq_len throws out_of_range");

        cache.reset();
        expect(cache.length() == 0, "reset zeroes length");

        bool threw_dim_mismatch = false;
        try
        {
            Tensor bad_k({1, 3}, {0.0f, 0.0f, 0.0f});
            Tensor bad_v({1, 2}, {0.0f, 0.0f});
            cache.append(bad_k, bad_v);
        }
        catch (const std::invalid_argument &)
        {
            threw_dim_mismatch = true;
        }
        expect(threw_dim_mismatch, "append with mismatched hidden_dim throws");
    }

    // Feeds the two tokens of `input` through attention one at a time against a shared
    // cache and checks the result matches the single-shot no-cache forward() over both
    // tokens at once - the core correctness property of a KV cache.
    void test_attention_incremental_matches_full_recompute()
    {
        const std::size_t hidden_dim = 2;
        const std::size_t num_heads = 1;
        MultiHeadAttention attn(hidden_dim, num_heads, /*causal=*/true, 10000.0f, 8,
                                 identity_weights(hidden_dim), {},
                                 identity_weights(hidden_dim), {},
                                 identity_weights(hidden_dim), {},
                                 identity_weights(hidden_dim), {});

        Tensor input({2, 2}, {1.0f, 0.0f,
                              0.0f, 1.0f});
        Tensor full_output = attn.forward(input);

        KvCache cache(4, hidden_dim);
        Tensor row0({1, 2}, {1.0f, 0.0f});
        Tensor row1({1, 2}, {0.0f, 1.0f});

        Tensor step0_output = attn.forward(row0, cache);
        expect(cache.length() == 1, "cache length after first decode step");
        expect_close(step0_output.at({0, 0}), full_output.at({0, 0}), "incremental step 0 matches full recompute (dim 0)");
        expect_close(step0_output.at({0, 1}), full_output.at({0, 1}), "incremental step 0 matches full recompute (dim 1)");

        Tensor step1_output = attn.forward(row1, cache);
        expect(cache.length() == 2, "cache length after second decode step");
        expect_close(step1_output.at({0, 0}), full_output.at({1, 0}), "incremental step 1 matches full recompute (dim 0)");
        expect_close(step1_output.at({0, 1}), full_output.at({1, 1}), "incremental step 1 matches full recompute (dim 1)");
    }

    // A single prefill call over the whole sequence should match the no-cache forward()
    // over the same sequence too, not just token-at-a-time decode.
    void test_attention_prefill_matches_full_recompute()
    {
        const std::size_t hidden_dim = 2;
        const std::size_t num_heads = 1;
        MultiHeadAttention attn(hidden_dim, num_heads, /*causal=*/true, 10000.0f, 8,
                                 identity_weights(hidden_dim), {},
                                 identity_weights(hidden_dim), {},
                                 identity_weights(hidden_dim), {},
                                 identity_weights(hidden_dim), {});

        Tensor input({2, 2}, {1.0f, 0.0f,
                              0.0f, 1.0f});
        Tensor full_output = attn.forward(input);

        KvCache cache(4, hidden_dim);
        Tensor prefill_output = attn.forward(input, cache);
        expect(cache.length() == 2, "cache length after prefill");
        for (std::size_t row = 0; row < 2; ++row)
        {
            for (std::size_t col = 0; col < 2; ++col)
            {
                expect_close(prefill_output.at({row, col}), full_output.at({row, col}),
                             "prefill output matches full recompute");
            }
        }
    }

} // namespace

int main()
{
    test_kv_cache_basics();
    test_attention_incremental_matches_full_recompute();
    test_attention_prefill_matches_full_recompute();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All kv cache tests passed" << std::endl;
    return 0;
}
