#pragma once

#include <cstddef>
#include <vector>

#include "layers/linear.h"
#include "tensor/tensor.h"

namespace mini_inference::layers
{

    // SwiGLU feed-forward block (LLaMA-style MLP): down_proj(silu(gate_proj(x)) * up_proj(x)).
    // gate_proj and up_proj both map hidden_dim -> intermediate_dim; down_proj maps back
    // intermediate_dim -> hidden_dim. The SiLU gate lets the network modulate which
    // intermediate features pass through, in place of a plain ReLU/GELU MLP.
    class SwiGLU
    {
    public:
        SwiGLU(std::size_t hidden_dim, std::size_t intermediate_dim,
               std::vector<float> gate_weights = {}, std::vector<float> gate_bias = {},
               std::vector<float> up_weights = {}, std::vector<float> up_bias = {},
               std::vector<float> down_weights = {}, std::vector<float> down_bias = {});

        std::size_t hidden_dim() const;
        std::size_t intermediate_dim() const;

        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input) const;

    private:
        std::size_t hidden_dim_{0};
        std::size_t intermediate_dim_{0};

        Linear gate_proj_;
        Linear up_proj_;
        Linear down_proj_;
    };

} // namespace mini_inference::layers
