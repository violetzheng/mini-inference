#pragma once

#include <cstddef>
#include <vector>

#include "tensor/tensor.h"

namespace mini_inference::layers
{

    class Linear
    {
    public:
        Linear(std::size_t in_features, std::size_t out_features,
               std::vector<float> weights = {}, std::vector<float> bias = {});

        const std::vector<float> &weights() const;
        const std::vector<float> &bias() const;

        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input) const;

    private:
        std::size_t in_features_{0};
        std::size_t out_features_{0};
        std::vector<float> weights_{};
        std::vector<float> bias_{};
    };

} // namespace mini_inference::layers
