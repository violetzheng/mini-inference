#include "layers/rope.h"

#include <cmath>
#include <stdexcept>

namespace mini_inference::layers
{

    RoPE::RoPE(std::size_t dim, float theta, std::size_t max_position_embeddings)
        : dim_(dim), theta_(theta), max_position_embeddings_(max_position_embeddings)
    {
        if (dim_ == 0 || dim_ % 2 != 0)
        {
            throw std::invalid_argument("RoPE dimension must be a positive even number");
        }
        if (max_position_embeddings_ == 0)
        {
            throw std::invalid_argument("max_position_embeddings must be greater than zero");
        }

        build_cache();
    }

    std::size_t RoPE::dim() const
    {
        return dim_;
    }

    float RoPE::theta() const
    {
        return theta_;
    }

    std::size_t RoPE::max_position_embeddings() const
    {
        return max_position_embeddings_;
    }

    void RoPE::build_cache()
    {
        const std::size_t half_dim = dim_ / 2;

        inv_freq_.resize(half_dim);
        for (std::size_t i = 0; i < half_dim; ++i)
        {
            inv_freq_[i] = 1.0f / std::pow(theta_, static_cast<float>(2 * i) / static_cast<float>(dim_));
        }

        cos_cache_.assign(max_position_embeddings_ * half_dim, 0.0f);
        sin_cache_.assign(max_position_embeddings_ * half_dim, 0.0f);

        for (std::size_t pos = 0; pos < max_position_embeddings_; ++pos)
        {
            for (std::size_t i = 0; i < half_dim; ++i)
            {
                const float angle = static_cast<float>(pos) * inv_freq_[i];
                cos_cache_[pos * half_dim + i] = std::cos(angle);
                sin_cache_[pos * half_dim + i] = std::sin(angle);
            }
        }
    }

    mini_inference::tensor::Tensor RoPE::forward(const mini_inference::tensor::Tensor &input,
                                                  std::size_t position_offset) const
    {
        if (input.rank() != 2)
        {
            throw std::invalid_argument("RoPE expects a 2D input tensor");
        }

        const std::size_t seq_len = input.shape()[0];
        std::vector<std::size_t> positions(seq_len);
        for (std::size_t i = 0; i < seq_len; ++i)
        {
            positions[i] = position_offset + i;
        }

        return forward(input, positions);
    }

    mini_inference::tensor::Tensor RoPE::forward(const mini_inference::tensor::Tensor &input,
                                                  const std::vector<std::size_t> &positions) const
    {
        if (input.rank() != 2)
        {
            throw std::invalid_argument("RoPE expects a 2D input tensor");
        }

        const auto &shape = input.shape();
        const std::size_t seq_len = shape[0];
        const std::size_t dim = shape[1];

        if (dim != dim_)
        {
            throw std::invalid_argument("input feature count does not match RoPE dimension");
        }
        if (positions.size() != seq_len)
        {
            throw std::invalid_argument("positions size must match input sequence length");
        }

        const std::size_t half_dim = dim_ / 2;
        std::vector<float> output_values(seq_len * dim_, 0.0f);

        for (std::size_t row = 0; row < seq_len; ++row)
        {
            const std::size_t pos = positions[row];
            if (pos >= max_position_embeddings_)
            {
                throw std::invalid_argument("position exceeds RoPE max_position_embeddings");
            }

            for (std::size_t i = 0; i < half_dim; ++i)
            {
                const float cos_value = cos_cache_[pos * half_dim + i];
                const float sin_value = sin_cache_[pos * half_dim + i];

                const float x_even = input.at({row, 2 * i});
                const float x_odd = input.at({row, 2 * i + 1});

                output_values[row * dim_ + 2 * i] = x_even * cos_value - x_odd * sin_value;
                output_values[row * dim_ + 2 * i + 1] = x_even * sin_value + x_odd * cos_value;
            }
        }

        return mini_inference::tensor::Tensor({seq_len, dim_}, std::move(output_values));
    }

} // namespace mini_inference::layers
