#pragma once

#include <cstddef>

#include "layers/attention.h"
#include "layers/rms_norm.h"
#include "layers/swiglu.h"
#include "tensor/tensor.h"

namespace mini_inference::layers
{

    // Standard pre-norm LLaMA-style decoder block: x = x + attention(rms_norm(x));
    // x = x + swiglu(rms_norm(x)). Composes already-configured sub-layers rather than
    // raw weights, since its inputs are other layer objects, not flat tensors.
    class TransformerBlock
    {
    public:
        TransformerBlock(RmsNorm attn_norm, MultiHeadAttention attention,
                          RmsNorm ffn_norm, SwiGLU ffn);

        std::size_t hidden_dim() const;
        std::size_t num_heads() const;
        std::size_t intermediate_dim() const;

        // position_offset is forwarded to attention/RoPE unchanged, the same future
        // KV-cache extension point RoPE/MultiHeadAttention already document.
        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input,
                                                std::size_t position_offset = 0) const;

        // Cache-aware path: forwards to attention_.forward(normed_input, cache).
        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input,
                                                KvCache &cache) const;

    private:
        std::size_t hidden_dim_{0};
        std::size_t num_heads_{0};
        std::size_t intermediate_dim_{0};

        RmsNorm attn_norm_;
        MultiHeadAttention attention_;
        RmsNorm ffn_norm_;
        SwiGLU ffn_;
    };

} // namespace mini_inference::layers
