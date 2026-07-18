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

    std::vector<float> identity_weights(std::size_t n)
    {
        std::vector<float> weights(n * n, 0.0f);
        for (std::size_t i = 0; i < n; ++i)
        {
            weights[i * n + i] = 1.0f;
        }
        return weights;
    }

    // Bytes for one Q6_K block (210 bytes, 256 elements) that dequantizes to a uniform
    // value of exactly `scale` for every one of its 256 elements: ql=0x11 and qh=0xAA
    // repeating make every 6-bit quant re-center to 1 (see dequantize_block_q6_k's bit
    // layout), d=1.0, so v = d * scale * 1 = scale for every element.
    std::vector<std::uint8_t> uniform_q6_k_block_bytes(std::uint8_t scale)
    {
        std::vector<std::uint8_t> block;
        block.reserve(210);
        block.insert(block.end(), 128, 0x11); // ql
        block.insert(block.end(), 64, 0xAA);  // qh
        block.insert(block.end(), 16, scale); // scales (all sub-block scales equal)
        block.push_back(0x00);                // d = 1.0 (fp16), low byte
        block.push_back(0x3C);                // d = 1.0 (fp16), high byte
        return block;
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

    // --- grouped-query attention (head_count_kv != head_count) loads successfully,
    // with attn_k/attn_v shaped [hidden_dim, num_kv_heads*head_dim] rather than
    // [hidden_dim, hidden_dim] ---
    {
        GgufBufferBuilder builder;
        builder.add_string_kv("general.architecture", "llama");
        builder.add_uint32_kv("llama.embedding_length", 4);
        builder.add_uint32_kv("llama.block_count", 1);
        builder.add_uint32_kv("llama.feed_forward_length", 1);
        builder.add_uint32_kv("llama.attention.head_count", 2);
        builder.add_uint32_kv("llama.attention.head_count_kv", 1);
        builder.add_float32_kv("llama.attention.layer_norm_rms_epsilon", 1e-5f);
        builder.add_float32_kv("llama.rope.freq_base", 10000.0f);
        builder.add_uint32_kv("llama.context_length", 8);

        builder.add_tensor_f32("token_embd.weight", {4, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});

        builder.add_tensor_f32("blk.0.attn_norm.weight", {4}, {1.0f, 1.0f, 1.0f, 1.0f});
        builder.add_tensor_f32("blk.0.attn_q.weight", {4, 4},
                                {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                                 0.0f, 1.0f});
        // kv_dim = num_kv_heads(1) * head_dim(2) = 2, narrower than hidden_dim.
        builder.add_tensor_f32("blk.0.attn_k.weight", {4, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        builder.add_tensor_f32("blk.0.attn_v.weight", {4, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f});
        builder.add_tensor_f32("blk.0.attn_output.weight", {4, 4},
                                {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                                 0.0f, 1.0f});

        builder.add_tensor_f32("blk.0.ffn_norm.weight", {4}, {1.0f, 1.0f, 1.0f, 1.0f});
        builder.add_tensor_f32("blk.0.ffn_gate.weight", {4, 1}, {1.0f, 0.0f, 0.0f, 0.0f});
        builder.add_tensor_f32("blk.0.ffn_up.weight", {4, 1}, {0.0f, 1.0f, 0.0f, 0.0f});
        builder.add_tensor_f32("blk.0.ffn_down.weight", {1, 4}, {1.0f, 1.0f, 1.0f, 1.0f});

        builder.add_tensor_f32("output_norm.weight", {4}, {1.0f, 1.0f, 1.0f, 1.0f});
        builder.add_tensor_f32("output.weight", {4, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f});

        GgufReader reader(builder.build());
        Model model = mini_inference::loader::build_model(reader);

        expect(model.hidden_dim() == 4, "gqa checkpoint has the expected hidden_dim");

        bool threw = false;
        Tensor output;
        try
        {
            output = model.forward({0});
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        expect(!threw, "gqa checkpoint's forward pass does not throw");
        expect(output.rank() == 2 && output.shape()[0] == 1 && output.shape()[1] == 2,
               "gqa checkpoint's output shape is [1, vocab_size]");
    }

    // --- token_embd.weight quantized as Q6_K (as real "Q4_K_M" GGUF files store it) ---
    {
        constexpr std::size_t kHiddenDim = 256; // must be a multiple of Q6_K's 256-element block size
        // eps=0 makes every intermediate value an exact power-of-two-friendly float
        // (sqrt(16)=4, sqrt(25)=5, ...), so the hand-derived check below needs no
        // floating-point tolerance slack from chained sqrt/divide rounding.
        constexpr float kEps = 0.0f;

        GgufBufferBuilder builder;
        builder.add_string_kv("general.architecture", "llama");
        builder.add_uint32_kv("llama.embedding_length", kHiddenDim);
        builder.add_uint32_kv("llama.block_count", 1);
        builder.add_uint32_kv("llama.feed_forward_length", 1);
        builder.add_uint32_kv("llama.attention.head_count", 1);
        builder.add_uint32_kv("llama.attention.head_count_kv", 1);
        builder.add_float32_kv("llama.attention.layer_norm_rms_epsilon", kEps);
        builder.add_float32_kv("llama.rope.freq_base", 10000.0f);
        builder.add_uint32_kv("llama.context_length", 8);

        // Row 0 (token id 0) dequantizes to a uniform 4.0 across all 256 features;
        // row 1 to a uniform 2.0.
        std::vector<std::uint8_t> embd_blocks = uniform_q6_k_block_bytes(4);
        const std::vector<std::uint8_t> row1 = uniform_q6_k_block_bytes(2);
        embd_blocks.insert(embd_blocks.end(), row1.begin(), row1.end());
        builder.add_tensor_q6_k_raw("token_embd.weight", {kHiddenDim, 2}, embd_blocks);

        builder.add_tensor_f32("blk.0.attn_norm.weight", {kHiddenDim}, std::vector<float>(kHiddenDim, 1.0f));
        builder.add_tensor_f32("blk.0.attn_q.weight", {kHiddenDim, kHiddenDim}, identity_weights(kHiddenDim));
        builder.add_tensor_f32("blk.0.attn_k.weight", {kHiddenDim, kHiddenDim}, identity_weights(kHiddenDim));
        builder.add_tensor_f32("blk.0.attn_v.weight", {kHiddenDim, kHiddenDim}, identity_weights(kHiddenDim));
        builder.add_tensor_f32("blk.0.attn_output.weight", {kHiddenDim, kHiddenDim}, identity_weights(kHiddenDim));

        builder.add_tensor_f32("blk.0.ffn_norm.weight", {kHiddenDim}, std::vector<float>(kHiddenDim, 1.0f));
        // Zero gate weights make silu(gate)*up == 0 regardless of up/down, so the FFN
        // contributes nothing to the residual - keeps the hand-derived check below simple.
        builder.add_tensor_f32("blk.0.ffn_gate.weight", {kHiddenDim, 1}, std::vector<float>(kHiddenDim, 0.0f));
        builder.add_tensor_f32("blk.0.ffn_up.weight", {kHiddenDim, 1}, std::vector<float>(kHiddenDim, 0.0f));
        builder.add_tensor_f32("blk.0.ffn_down.weight", {1, kHiddenDim}, std::vector<float>(kHiddenDim, 0.0f));

        builder.add_tensor_f32("output_norm.weight", {kHiddenDim}, std::vector<float>(kHiddenDim, 1.0f));
        // output.weight omitted: tied embeddings reuse the (dequantized) token_embd.weight.

        GgufReader reader(builder.build());
        Model model = mini_inference::loader::build_model(reader);

        expect(model.hidden_dim() == kHiddenDim, "q6_k embedding checkpoint has the expected hidden_dim");

        const Tensor output = model.forward({0});
        expect(output.rank() == 2 && output.shape()[0] == 1 && output.shape()[1] == 2,
               "q6_k embedding checkpoint's output shape is [1, vocab_size]");

        // Hand-derived: with a uniform hidden vector of value v, RmsNorm's mean-of-squares
        // is exactly v^2 (the /dim cancels for a uniform vector), attention with identity
        // Q/K/V/O and a single causal position is a no-op (softmax weight 1.0), and the
        // zeroed FFN contributes nothing to the residual.
        const float x = 4.0f; // token_embd row 0's dequantized value
        const float rms1 = std::sqrt(x * x + kEps);
        const float n1 = x / rms1;
        const float r1 = x + n1;
        const float rms3 = std::sqrt(r1 * r1 + kEps);
        const float final_val = r1 / rms3;

        expect_close(output.at({0, 0}), static_cast<float>(kHiddenDim) * final_val * 4.0f,
                     "q6_k embedding logit 0 (tied to row 0, dequantized value 4.0)");
        expect_close(output.at({0, 1}), static_cast<float>(kHiddenDim) * final_val * 2.0f,
                     "q6_k embedding logit 1 (tied to row 1, dequantized value 2.0)");
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
