#include "loader/gguf_model_loader.h"
#include "gguf_test_helpers.h"
#include "loader/gguf_reader.h"
#include "tensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using mini_inference::loader::GgufReader;
using mini_inference::model::Model;
using mini_inference::tensor::Tensor;
using mini_inference::tests::GgufBufferBuilder;

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

    // Same 1-layer fixture as test_model.cpp's make_embedding()/make_block()/make_lm_head().
    void add_llama_checkpoint(GgufBufferBuilder &builder, bool include_output_weight = true,
                               bool include_ffn_down_weight = true, bool attn_q_unsupported_type = false)
    {
        builder.add_string_kv("general.architecture", "llama");
        builder.add_uint32_kv("llama.embedding_length", 2);
        builder.add_uint32_kv("llama.block_count", 1);
        builder.add_uint32_kv("llama.feed_forward_length", 1);
        builder.add_uint32_kv("llama.attention.head_count", 1);
        builder.add_uint32_kv("llama.attention.head_count_kv", 1);
        builder.add_float32_kv("llama.attention.layer_norm_rms_epsilon", 1e-5f);
        builder.add_float32_kv("llama.rope.freq_base", 10000.0f);
        builder.add_uint32_kv("llama.context_length", 8);

        builder.add_tensor_f32("token_embd.weight", {2, 3}, {1.0f, 2.0f, 0.0f, 0.0f, -1.0f, -1.0f});

        builder.add_tensor_f32("blk.0.attn_norm.weight", {2}, {1.0f, 1.0f});
        if (attn_q_unsupported_type)
        {
            builder.add_tensor_unsupported_type("blk.0.attn_q.weight", {2, 2}, /*ggml_type=*/2, /*byte_count=*/16);
        }
        else
        {
            builder.add_tensor_f32("blk.0.attn_q.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
        }
        builder.add_tensor_f32("blk.0.attn_k.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
        builder.add_tensor_f32("blk.0.attn_v.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
        builder.add_tensor_f32("blk.0.attn_output.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});

        builder.add_tensor_f32("blk.0.ffn_norm.weight", {2}, {1.0f, 1.0f});
        builder.add_tensor_f32("blk.0.ffn_gate.weight", {2, 1}, {1.0f, 0.0f});
        builder.add_tensor_f32("blk.0.ffn_up.weight", {2, 1}, {0.0f, 1.0f});
        if (include_ffn_down_weight)
        {
            builder.add_tensor_f32("blk.0.ffn_down.weight", {1, 2}, {2.0f, -1.0f});
        }

        builder.add_tensor_f32("output_norm.weight", {2}, {1.0f, 1.0f});
        if (include_output_weight)
        {
            builder.add_tensor_f32("output.weight", {2, 3}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f});
        }
    }

    float silu(float x)
    {
        return x / (1.0f + std::exp(-x));
    }

    // test_model.cpp's hand-derived forward pass, up to the post-final-RmsNorm state.
    void compute_final_hidden_state(float &final0, float &final1)
    {
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
        final0 = block_out0 / final_rms;
        final1 = block_out1 / final_rms;
    }

} // namespace

int main()
{
    float final0 = 0.0f;
    float final1 = 0.0f;
    compute_final_hidden_state(final0, final1);

    // --- golden path: matches test_model.cpp's hand-derived logits exactly ---
    {
        GgufBufferBuilder builder;
        add_llama_checkpoint(builder);
        GgufReader reader(builder.build());
        Model model = mini_inference::loader::build_model(reader);

        expect(model.vocab_size() == 3, "loaded model has the expected vocab_size");
        expect(model.hidden_dim() == 2, "loaded model has the expected hidden_dim");
        expect(model.num_layers() == 1, "loaded model has the expected num_layers");

        const Tensor output = model.forward({0});
        expect(output.rank() == 2 && output.shape()[0] == 1 && output.shape()[1] == 3, "output shape is [1, vocab_size]");
        expect_close(output.at({0, 0}), final0, "logit 0 matches test_model.cpp's closed-form derivation");
        expect_close(output.at({0, 1}), final1, "logit 1 matches test_model.cpp's closed-form derivation");
        expect_close(output.at({0, 2}), final0 + final1, "logit 2 matches test_model.cpp's closed-form derivation");
    }

    // --- tied embeddings: output.weight omitted, lm_head reuses token_embd.weight ---
    {
        GgufBufferBuilder builder;
        add_llama_checkpoint(builder, /*include_output_weight=*/false);
        GgufReader reader(builder.build());
        Model model = mini_inference::loader::build_model(reader);

        const Tensor output = model.forward({0});
        expect_close(output.at({0, 0}), final0 * 1.0f + final1 * 2.0f, "tied lm_head logit 0 uses the embedding row");
        expect_close(output.at({0, 1}), final0 * 0.0f + final1 * 0.0f, "tied lm_head logit 1 uses the embedding row");
        expect_close(output.at({0, 2}), final0 * -1.0f + final1 * -1.0f, "tied lm_head logit 2 uses the embedding row");
    }

    // --- wrong architecture ---
    {
        GgufBufferBuilder builder;
        builder.add_string_kv("general.architecture", "gpt2");
        GgufReader reader(builder.build());

        bool threw = false;
        try
        {
            (void)mini_inference::loader::build_model(reader);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        expect(threw, "unsupported architecture throws invalid_argument");
    }

    // --- grouped-query attention (head_count_kv != head_count) ---
    {
        GgufBufferBuilder builder;
        builder.add_string_kv("general.architecture", "llama");
        builder.add_uint32_kv("llama.embedding_length", 2);
        builder.add_uint32_kv("llama.block_count", 1);
        builder.add_uint32_kv("llama.feed_forward_length", 1);
        builder.add_uint32_kv("llama.attention.head_count", 2);
        builder.add_uint32_kv("llama.attention.head_count_kv", 1);
        GgufReader reader(builder.build());

        bool threw = false;
        try
        {
            (void)mini_inference::loader::build_model(reader);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        expect(threw, "grouped-query attention checkpoint throws invalid_argument");
    }

    // --- unsupported quantization type ---
    {
        GgufBufferBuilder builder;
        add_llama_checkpoint(builder, /*include_output_weight=*/true, /*include_ffn_down_weight=*/true,
                              /*attn_q_unsupported_type=*/true);
        GgufReader reader(builder.build());

        bool threw = false;
        try
        {
            (void)mini_inference::loader::build_model(reader);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        expect(threw, "unsupported ggml tensor type throws invalid_argument");
    }

    // --- missing required tensor ---
    {
        GgufBufferBuilder builder;
        add_llama_checkpoint(builder, /*include_output_weight=*/true, /*include_ffn_down_weight=*/false);
        GgufReader reader(builder.build());

        bool threw = false;
        try
        {
            (void)mini_inference::loader::build_model(reader);
        }
        catch (const std::out_of_range &)
        {
            threw = true;
        }
        expect(threw, "missing required tensor propagates out_of_range naming it");
    }

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All gguf_model_loader tests passed" << std::endl;
    return 0;
}
