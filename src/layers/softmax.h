#pragma once

#include <cstddef>
#include <vector>

#include "tensor/tensor.h"

namespace mini_inference::layers
{

    class Softmax
    {
    public:
        Softmax(std::size_t dim = 1);

        std::size_t dim() const;

        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input) const;

    private:
        std::size_t dim_{1};
    };

} // namespace mini_inference::layers
