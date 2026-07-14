#include "layers/linear.h"

#include <cassert>
#include <stdexcept>

namespace mini_inference::layers
{

    namespace
    {

        std::size_t expected_weight_count(std::size_t in_features, std::size_t out_features)
        {
            return in_features * out_features;
        }

    } // namespace

    Linear::Linear(std::size_t in_features, std::size_t out_features,
                   std::vector<float> weights, std::vector<float> bias)
        : in_features_(in_features), out_features_(out_features), weights_(std::move(weights)), bias_(std::move(bias))
    {
        if (in_features_ == 0 || out_features_ == 0)
        {
            throw std::invalid_argument("linear layer dimensions must be greater than zero");
        }

        if (weights_.empty())
        {
            weights_.assign(expected_weight_count(in_features_, out_features_), 0.0f);
        }
        else if (weights_.size() != expected_weight_count(in_features_, out_features_))
        {
            throw std::invalid_argument("weight count does not match layer dimensions");
        }

        if (bias_.empty())
        {
            bias_.assign(out_features_, 0.0f);
        }
        else if (bias_.size() != out_features_)
        {
            throw std::invalid_argument("bias count does not match output size");
        }
    }

    const std::vector<float> &Linear::weights() const
    {
        return weights_;
    }

    const std::vector<float> &Linear::bias() const
    {
        return bias_;
    }

    std::size_t Linear::in_features() const
    {
        return in_features_;
    }

    std::size_t Linear::out_features() const
    {
        return out_features_;
    }

    mini_inference::tensor::Tensor Linear::forward(const mini_inference::tensor::Tensor &input) const
    {
        if (input.rank() != 2)
        {
            throw std::invalid_argument("linear layer expects a 2D input tensor");
        }

        const auto &shape = input.shape();
        if (shape[1] != in_features_)
        {
            throw std::invalid_argument("input feature count does not match layer input size");
        }

        const std::size_t batch_size = shape[0];
        std::vector<float> output_values(batch_size * out_features_, 0.0f);

        for (std::size_t batch = 0; batch < batch_size; ++batch)
        {
            for (std::size_t out_idx = 0; out_idx < out_features_; ++out_idx)
            {
                float sum = bias_[out_idx];
                for (std::size_t in_idx = 0; in_idx < in_features_; ++in_idx)
                {
                    const std::size_t flat_input_index = batch * in_features_ + in_idx;
                    const std::size_t weight_index = out_idx * in_features_ + in_idx;
                    sum += input.at(flat_input_index) * weights_[weight_index];
                }
                output_values[batch * out_features_ + out_idx] = sum;
            }
        }

        return mini_inference::tensor::Tensor({batch_size, out_features_}, std::move(output_values));
    }

} // namespace mini_inference::layers
