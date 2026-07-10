#include "layers/rms_norm.h"

#include <cmath>
#include <stdexcept>

namespace mini_inference::layers
{
    RmsNorm::RmsNorm(std::size_t dim, float eps, std::vector<float> gamma)
        : dim_(dim), eps_(eps), gamma_(std::move(gamma))
    {
        if (dim_ == 0)
        {
            throw std::invalid_argument("rms norm dimension must be greater than zero");
        }

        if (gamma_.empty())
        {
            gamma_.assign(dim_, 1.0f);
        }
        else if (gamma_.size() != dim_)
        {
            throw std::invalid_argument("gamma size does not match rms norm dimension");
        }
    }

    std::size_t RmsNorm::dim() const
    {
        return dim_;
    }

    float RmsNorm::eps() const
    {
        return eps_;
    }

    const std::vector<float> &RmsNorm::gamma() const
    {
        return gamma_;
    }

    mini_inference::tensor::Tensor RmsNorm::forward(const mini_inference::tensor::Tensor &input) const
    {
        if (input.rank() != 2)
        {
            throw std::invalid_argument("rms norm expects a 2D input tensor");
        }

        const auto &shape = input.shape();
        if (shape[1] != dim_)
        {
            throw std::invalid_argument("input feature count does not match rms norm dimension");
        }

        const std::size_t batch_size = shape[0];
        std::vector<float> output_values(batch_size * dim_, 0.0f);

        for (std::size_t batch = 0; batch < batch_size; ++batch)
        {
            float sum_squares = 0.0f;
            for (std::size_t feature = 0; feature < dim_; ++feature)
            {
                const float value = input.at({batch, feature});
                sum_squares += value * value;
            }

            const float rms = std::sqrt(sum_squares / static_cast<float>(dim_) + eps_);
            for (std::size_t feature = 0; feature < dim_; ++feature)
            {
                output_values[batch * dim_ + feature] = (input.at({batch, feature}) / rms) * gamma_[feature];
            }
        }

        return mini_inference::tensor::Tensor({batch_size, dim_}, std::move(output_values));
    }
}