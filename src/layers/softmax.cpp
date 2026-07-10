#include "layers/softmax.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace mini_inference::layers
{

    Softmax::Softmax(std::size_t dim) : dim_(dim)
    {
        if (dim_ == 0)
        {
            throw std::invalid_argument("softmax dimension must be greater than zero");
        }
    }

    std::size_t Softmax::dim() const
    {
        return dim_;
    }

    mini_inference::tensor::Tensor Softmax::forward(const mini_inference::tensor::Tensor &input) const
    {
        if (input.rank() != 2)
        {
            throw std::invalid_argument("softmax expects a 2D input tensor");
        }

        const auto &shape = input.shape();
        if (shape.size() != 2)
        {
            throw std::invalid_argument("softmax expects a 2D input tensor");
        }

        const std::size_t rows = shape[0];
        const std::size_t cols = shape[1];
        if (dim_ == 0 || dim_ > 1)
        {
            throw std::invalid_argument("softmax currently supports only the last dimension");
        }

        std::vector<float> output_values(rows * cols, 0.0f);

        for (std::size_t row = 0; row < rows; ++row)
        {
            float max_value = input.at({row, 0});
            for (std::size_t col = 1; col < cols; ++col)
            {
                max_value = std::max(max_value, input.at({row, col}));
            }

            float sum_exp = 0.0f;
            for (std::size_t col = 0; col < cols; ++col)
            {
                const float value = std::exp(input.at({row, col}) - max_value);
                output_values[row * cols + col] = value;
                sum_exp += value;
            }

            for (std::size_t col = 0; col < cols; ++col)
            {
                output_values[row * cols + col] /= sum_exp;
            }
        }

        return mini_inference::tensor::Tensor({rows, cols}, std::move(output_values));
    }

} // namespace mini_inference::layers
