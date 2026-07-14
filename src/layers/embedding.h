#pragma once

#include <cstddef>
#include <vector>

#include "tensor/tensor.h"

namespace mini_inference::layers
{

    // Token embedding lookup table: one row of length hidden_dim per vocab entry.
    // Unlike every other layer, forward() consumes token ids, not a Tensor of floats —
    // it is the entry point that turns discrete tokens into the continuous
    // representation the rest of the network operates on.
    class Embedding
    {
    public:
        Embedding(std::size_t vocab_size, std::size_t hidden_dim, std::vector<float> weights = {});

        std::size_t vocab_size() const;
        std::size_t hidden_dim() const;
        const std::vector<float> &weights() const;

        mini_inference::tensor::Tensor forward(const std::vector<std::size_t> &token_ids) const;

    private:
        std::size_t vocab_size_{0};
        std::size_t hidden_dim_{0};
        std::vector<float> weights_{};
    };

} // namespace mini_inference::layers
