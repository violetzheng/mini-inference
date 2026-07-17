#include "layers/quantized_linear.h"

#include "tensor/parallel_for.h"
#include "tensor/simd_ops.h"

#include <stdexcept>

namespace mini_inference::layers
{

    QuantizedLinear::QuantizedLinear(mini_inference::tensor::QuantizedTensor weights, std::vector<float> bias)
        : weights_(std::move(weights)), bias_(std::move(bias))
    {
        if (bias_.empty())
        {
            bias_.assign(weights_.rows(), 0.0f);
        }
        else if (bias_.size() != weights_.rows())
        {
            throw std::invalid_argument("bias count does not match layer out_features");
        }
    }

    std::size_t QuantizedLinear::in_features() const
    {
        return weights_.cols();
    }

    std::size_t QuantizedLinear::out_features() const
    {
        return weights_.rows();
    }

    mini_inference::tensor::Tensor QuantizedLinear::forward(const mini_inference::tensor::Tensor &input) const
    {
        if (input.rank() != 2)
        {
            throw std::invalid_argument("quantized linear layer expects a 2D input tensor");
        }

        const auto &shape = input.shape();
        const std::size_t in_features_ = weights_.cols();
        const std::size_t out_features_ = weights_.rows();
        if (shape[1] != in_features_)
        {
            throw std::invalid_argument("input feature count does not match layer input size");
        }

        const std::size_t batch_size = shape[0];
        std::vector<float> output_values(batch_size * out_features_, 0.0f);

        mini_inference::tensor::parallel_for(0, out_features_, [&](std::size_t out_begin, std::size_t out_end) {
            std::vector<float> row_buffer; // scratch space local to this thread's chunk
            for (std::size_t out_idx = out_begin; out_idx < out_end; ++out_idx)
            {
                weights_.dequantize_row(out_idx, row_buffer);
                for (std::size_t batch = 0; batch < batch_size; ++batch)
                {
                    const float *input_row = input.values().data() + batch * in_features_;
                    output_values[batch * out_features_ + out_idx] =
                        bias_[out_idx] + mini_inference::tensor::dot_product_f32(input_row, row_buffer.data(), in_features_);
                }
            }
        });

        return mini_inference::tensor::Tensor({batch_size, out_features_}, std::move(output_values));
    }

} // namespace mini_inference::layers
