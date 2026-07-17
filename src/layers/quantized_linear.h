#pragma once

#include <cstddef>
#include <vector>

#include "tensor/quantized_tensor.h"
#include "tensor/tensor.h"

namespace mini_inference::layers
{

    // A linear projection whose weights stay stored in a packed quantized format
    // (see mini_inference::tensor::QuantizedTensor) instead of float32. Mirrors
    // Linear's public interface so it can be used as a drop-in alternative wherever
    // a projection's weights are quantized on disk. Bias stays float32 (biases are
    // tiny and GGUF checkpoints keep them unquantized).
    class QuantizedLinear
    {
    public:
        explicit QuantizedLinear(mini_inference::tensor::QuantizedTensor weights,
                                  std::vector<float> bias = {});

        std::size_t in_features() const;
        std::size_t out_features() const;

        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input) const;

    private:
        mini_inference::tensor::QuantizedTensor weights_;
        std::vector<float> bias_;
    };

} // namespace mini_inference::layers
