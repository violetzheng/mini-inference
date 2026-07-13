#pragma once

#include <cstddef>
#include <vector>

#include "layers/linear.h"
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

        std::size_t hidden_dim() const;
        std::size_t num_heads() const;
        std::size_t head_dim() const;
        bool causal() const;

        // Runs self-attention over `input` (shape [seq_len, hidden_dim]). `position_offset`
        // is forwarded to RoPE so queries/keys are rotated as if the sequence starts at that
        // position, e.g. once a KV cache is added later for resuming mid-sequence decoding.
        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input,
                                                std::size_t position_offset = 0) const;

    private:
        std::size_t hidden_dim_{0};
        std::size_t num_heads_{0};
        std::size_t head_dim_{0};
        bool causal_{true};

        Linear q_proj_;
        Linear k_proj_;
        Linear v_proj_;
        Linear o_proj_;
        RoPE rope_;
        Softmax softmax_;
    };

} // namespace mini_inference::layers
