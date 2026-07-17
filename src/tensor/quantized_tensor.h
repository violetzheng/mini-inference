#pragma once

#include <cstddef>
#include <vector>

#include "tensor/quant_blocks.h"
#include "tensor/tensor.h"

namespace mini_inference::tensor
{

    // A row-major [rows, cols] weight matrix stored packed per `format`, quantized
    // along `cols` (cols must be divisible by the format's block size). Weights stay
    // packed in memory; dequantization happens on demand, one block or one row at a
    // time, rather than eagerly expanding the whole matrix to float32 at load time.
    class QuantizedTensor
    {
    public:
        QuantizedTensor(QuantFormat format, std::size_t rows, std::size_t cols,
                         std::vector<std::byte> blocks);

        QuantFormat format() const;
        std::size_t rows() const;
        std::size_t cols() const;

        // Dequantizes the whole matrix into a float32 Tensor of shape [rows, cols].
        Tensor dequantize() const;

        // Dequantizes a single row into `out`, resizing it to cols() first.
        void dequantize_row(std::size_t row, std::vector<float> &out) const;

    private:
        QuantFormat format_;
        std::size_t rows_{0};
        std::size_t cols_{0};
        std::size_t blocks_per_row_{0};
        std::vector<std::byte> blocks_{};
    };

} // namespace mini_inference::tensor
