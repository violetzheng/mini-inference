#include "layers/embedding.h"

#include <stdexcept>

namespace mini_inference::layers
{

    namespace
    {

        std::size_t expected_weight_count(std::size_t vocab_size, std::size_t hidden_dim)
        {
            return vocab_size * hidden_dim;
        }

    } // namespace

    Embedding::Embedding(std::size_t vocab_size, std::size_t hidden_dim, std::vector<float> weights)
        : vocab_size_(vocab_size), hidden_dim_(hidden_dim), weights_(std::move(weights))
    {
        if (vocab_size_ == 0 || hidden_dim_ == 0)
        {
            throw std::invalid_argument("embedding dimensions must be greater than zero");
        }

        if (weights_.empty())
        {
            weights_.assign(expected_weight_count(vocab_size_, hidden_dim_), 0.0f);
        }
        else if (weights_.size() != expected_weight_count(vocab_size_, hidden_dim_))
        {
            throw std::invalid_argument("weight count does not match embedding dimensions");
        }
    }

    std::size_t Embedding::vocab_size() const
    {
        return vocab_size_;
    }

    std::size_t Embedding::hidden_dim() const
    {
        return hidden_dim_;
    }

    const std::vector<float> &Embedding::weights() const
    {
        return weights_;
    }

    mini_inference::tensor::Tensor Embedding::forward(const std::vector<std::size_t> &token_ids) const
    {
        std::vector<float> output_values(token_ids.size() * hidden_dim_);

        for (std::size_t i = 0; i < token_ids.size(); ++i)
        {
            const std::size_t token_id = token_ids[i];
            if (token_id >= vocab_size_)
            {
                throw std::invalid_argument("token id out of range");
            }

            for (std::size_t feature = 0; feature < hidden_dim_; ++feature)
            {
                output_values[i * hidden_dim_ + feature] = weights_[token_id * hidden_dim_ + feature];
            }
        }

        return mini_inference::tensor::Tensor({token_ids.size(), hidden_dim_}, std::move(output_values));
    }

} // namespace mini_inference::layers
