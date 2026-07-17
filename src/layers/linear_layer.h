#pragma once

#include <cstddef>
#include <variant>

#include "layers/linear.h"
#include "layers/quantized_linear.h"
#include "tensor/tensor.h"

namespace mini_inference::layers
{

    // A projection whose weights are either plain float32 (Linear) or packed
    // quantized (QuantizedLinear). Lets a single checkpoint mix float and quantized
    // weights per-tensor (as real GGUF checkpoints do) without any call site needing
    // to know which representation a given projection uses.
    using LinearLayer = std::variant<Linear, QuantizedLinear>;

    mini_inference::tensor::Tensor forward(const LinearLayer &layer, const mini_inference::tensor::Tensor &input);
    std::size_t in_features(const LinearLayer &layer);
    std::size_t out_features(const LinearLayer &layer);

} // namespace mini_inference::layers
