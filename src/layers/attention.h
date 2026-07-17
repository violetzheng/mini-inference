#pragma once

#include <cstddef>
#include <vector>

#include "layers/kv_cache.h"
#include "layers/linear.h"
#include "layers/linear_layer.h"
#include "layers/rope.h"
#include "layers/softmax.h"
#include "tensor/tensor.h"

namespace mini_inference::layers
{

    // Multi-head self-attention, no KV cache: every call recomputes queries, keys
    // and values for the full input sequence. Q/K are rotated per-head with RoPE,
    // attention is scaled dot-product with an optional causal mask, and head
    // outputs are concatenated and passed through an output projection.
    class MultiHeadAttention
    {
    public:
        MultiHeadAttention(std::size_t hidden_dim, std::size_t num_heads, bool causal = true,
                            float rope_theta = 10000.0f, std::size_t max_position_embeddings = 2048,
                            std::vector<float> q_weights = {}, std::vector<float> q_bias = {},
                            std::vector<float> k_weights = {}, std::vector<float> k_bias = {},
                            std::vector<float> v_weights = {}, std::vector<float> v_bias = {},
                            std::vector<float> o_weights = {}, std::vector<float> o_bias = {});

        // Cache-friendly / quantization-friendly overload: caller supplies already-built
        // projections (each independently Linear or QuantizedLinear), e.g. from a GGUF
        // loader that decides per-tensor whether a projection's weights are quantized.
        MultiHeadAttention(std::size_t hidden_dim, std::size_t num_heads, bool causal,
                            float rope_theta, std::size_t max_position_embeddings,
                            LinearLayer q_proj, LinearLayer k_proj,
                            LinearLayer v_proj, LinearLayer o_proj);

        std::size_t hidden_dim() const;
        std::size_t num_heads() const;
        std::size_t head_dim() const;
        bool causal() const;

        // Runs self-attention over `input` (shape [seq_len, hidden_dim]). `position_offset`
        // is forwarded to RoPE so queries/keys are rotated as if the sequence starts at that
        // position, e.g. once a KV cache is added later for resuming mid-sequence decoding.
        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input,
                                                std::size_t position_offset = 0) const;

        // Cache-aware path: appends this step's rotated keys and raw values into `cache`
        // (at position cache.length()), then attends `input`'s queries against every
        // position cached so far, not just `input` itself. Used for prefill (input holds
        // the whole prompt, cache starts empty) and single-token decode (input is one row,
        // cache already holds every earlier position) alike.
        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input,
                                                KvCache &cache) const;

    private:
        std::size_t hidden_dim_{0};
        std::size_t num_heads_{0};
        std::size_t head_dim_{0};
        bool causal_{true};

        LinearLayer q_proj_;
        LinearLayer k_proj_;
        LinearLayer v_proj_;
        LinearLayer o_proj_;
        RoPE rope_;
        Softmax softmax_;
    };

} // namespace mini_inference::layers
