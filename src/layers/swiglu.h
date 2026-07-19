#pragma once

#include <cstddef>
#include <vector>

#include "layers/linear.h"
#include "layers/linear_layer.h"
#include "tensor/tensor.h"

namespace mini_inference::layers
{

    // kSilu: SwiGLU (LLaMA/Qwen2). kGelu: GeGLU (Gemma), tanh-approximated GELU.
    enum class GateActivation
    {
        kSilu,
        kGelu,
    };

    // Gated feed-forward block: down_proj(act(gate_proj(x)) * up_proj(x)). gate_proj/up_proj
    // map hidden_dim -> intermediate_dim; down_proj maps back.
    class SwiGLU
    {
    public:
        SwiGLU(std::size_t hidden_dim, std::size_t intermediate_dim,
               std::vector<float> gate_weights = {}, std::vector<float> gate_bias = {},
               std::vector<float> up_weights = {}, std::vector<float> up_bias = {},
               std::vector<float> down_weights = {}, std::vector<float> down_bias = {},
               GateActivation activation = GateActivation::kSilu);

        // Quantization-friendly overload: caller supplies already-built projections.
        SwiGLU(std::size_t hidden_dim, std::size_t intermediate_dim,
               LinearLayer gate_proj, LinearLayer up_proj, LinearLayer down_proj,
               GateActivation activation = GateActivation::kSilu);

        std::size_t hidden_dim() const;
        std::size_t intermediate_dim() const;

        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input) const;

    private:
        std::size_t hidden_dim_{0};
        std::size_t intermediate_dim_{0};

        LinearLayer gate_proj_;
        LinearLayer up_proj_;
        LinearLayer down_proj_;
        GateActivation activation_{GateActivation::kSilu};
    };

} // namespace mini_inference::layers
