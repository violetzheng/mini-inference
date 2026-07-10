#pragma once

#include <cstddef>
#include <vector>

#include "tensor/tensor.h"

namespace mini_inference::layers
{

    class RmsNorm
    {
    public:
        RmsNorm(std::size_t dim, float eps = 1e-5f, std::vector<float> gamma = {});

        std::size_t dim() const;
        float eps() const;
        const std::vector<float> &gamma() const;

        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input) const;

    private:
        std::size_t dim_{0};
        float eps_{1e-5f};
        std::vector<float> gamma_{};
    };

} // namespace mini_inference::layers
