#include "layers/swiglu.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mini_inference::layers
{

    namespace
    {

        std::size_t validate_dim(std::size_t dim, const char *name)
        {
            if (dim == 0)
            {
                throw std::invalid_argument(std::string("swiglu ") + name + " must be greater than zero");
            }
            return dim;
        }

        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
        // One branch-free, contiguous pass over gate, up, out to reduce cache misses, easier to vectorize
        void fused_silu_mul(const float *__restrict gate, const float *__restrict up,
                             float *__restrict out, std::size_t n)
        {
            assert(gate != nullptr && up != nullptr && out != nullptr);

#pragma clang loop vectorize(enable) interleave(enable)
            for (std::size_t i = 0; i < n; ++i)
            {
                const float g = gate[i];
                const float sigmoid_g = 1.0f / (1.0f + std::exp(-g));
                out[i] = g * sigmoid_g * up[i];
            }
        }

        // Tanh approximation of GELU (Gemma's "gelu_pytorch_tanh").
        void fused_gelu_mul(const float *__restrict gate, const float *__restrict up,
                             float *__restrict out, std::size_t n)
        {
            assert(gate != nullptr && up != nullptr && out != nullptr);

            constexpr float kSqrt2OverPi = 0.7978845608028654f;
#pragma clang loop vectorize(enable) interleave(enable)
            for (std::size_t i = 0; i < n; ++i)
            {
                const float g = gate[i];
                const float inner = kSqrt2OverPi * (g + 0.044715f * g * g * g);
                const float gelu_g = 0.5f * g * (1.0f + std::tanh(inner));
                out[i] = gelu_g * up[i];
            }
        }

    } // namespace

    SwiGLU::SwiGLU(std::size_t hidden_dim, std::size_t intermediate_dim,
                   std::vector<float> gate_weights, std::vector<float> gate_bias,
                   std::vector<float> up_weights, std::vector<float> up_bias,
                   std::vector<float> down_weights, std::vector<float> down_bias,
                   GateActivation activation)
        : hidden_dim_(validate_dim(hidden_dim, "hidden_dim")),
          intermediate_dim_(validate_dim(intermediate_dim, "intermediate_dim")),
          gate_proj_(Linear(hidden_dim_, intermediate_dim_, std::move(gate_weights), std::move(gate_bias))),
          up_proj_(Linear(hidden_dim_, intermediate_dim_, std::move(up_weights), std::move(up_bias))),
          down_proj_(Linear(intermediate_dim_, hidden_dim_, std::move(down_weights), std::move(down_bias))),
          activation_(activation)
    {
    }

    SwiGLU::SwiGLU(std::size_t hidden_dim, std::size_t intermediate_dim,
                   LinearLayer gate_proj, LinearLayer up_proj, LinearLayer down_proj,
                   GateActivation activation)
        : hidden_dim_(validate_dim(hidden_dim, "hidden_dim")),
          intermediate_dim_(validate_dim(intermediate_dim, "intermediate_dim")),
          gate_proj_(std::move(gate_proj)),
          up_proj_(std::move(up_proj)),
          down_proj_(std::move(down_proj)),
          activation_(activation)
    {
        if (mini_inference::layers::in_features(gate_proj_) != hidden_dim_ ||
            mini_inference::layers::out_features(gate_proj_) != intermediate_dim_ ||
            mini_inference::layers::in_features(up_proj_) != hidden_dim_ ||
            mini_inference::layers::out_features(up_proj_) != intermediate_dim_ ||
            mini_inference::layers::in_features(down_proj_) != intermediate_dim_ ||
            mini_inference::layers::out_features(down_proj_) != hidden_dim_)
        {
            throw std::invalid_argument("swiglu projection dimensions do not match hidden_dim/intermediate_dim");
        }
    }

    std::size_t SwiGLU::hidden_dim() const
    {
        return hidden_dim_;
    }

    std::size_t SwiGLU::intermediate_dim() const
    {
        return intermediate_dim_;
    }

    mini_inference::tensor::Tensor SwiGLU::forward(const mini_inference::tensor::Tensor &input) const
    {
        if (input.rank() != 2)
        {
            throw std::invalid_argument("swiglu layer expects a 2D input tensor");
        }

        const auto &shape = input.shape();
        if (shape[1] != hidden_dim_)
        {
            throw std::invalid_argument("input feature count does not match swiglu hidden_dim");
        }

        const mini_inference::tensor::Tensor gate = mini_inference::layers::forward(gate_proj_, input);
        const mini_inference::tensor::Tensor up = mini_inference::layers::forward(up_proj_, input);
        assert(gate.numel() == up.numel());

        std::vector<float> fused(gate.numel());
        if (activation_ == GateActivation::kGelu)
        {
            fused_gelu_mul(gate.values().data(), up.values().data(), fused.data(), fused.size());
        }
        else
        {
            fused_silu_mul(gate.values().data(), up.values().data(), fused.data(), fused.size());
        }

        const mini_inference::tensor::Tensor fused_tensor({shape[0], intermediate_dim_}, std::move(fused));
        return mini_inference::layers::forward(down_proj_, fused_tensor);
    }

} // namespace mini_inference::layers
