#include "tensor/quantized_tensor.h"

#include <algorithm>
#include <stdexcept>

namespace mini_inference::tensor
{

    QuantizedTensor::QuantizedTensor(QuantFormat format, std::size_t rows, std::size_t cols,
                                      std::vector<std::byte> blocks)
        : format_(format), rows_(rows), cols_(cols), blocks_(std::move(blocks))
    {
        if (rows_ == 0 || cols_ == 0)
        {
            throw std::invalid_argument("quantized tensor dimensions must be greater than zero");
        }

        const std::size_t block_elems = block_size(format_);
        if (cols_ % block_elems != 0)
        {
            throw std::invalid_argument("quantized tensor cols must be a multiple of the format's block size");
        }
        blocks_per_row_ = cols_ / block_elems;

        const std::size_t expected_bytes = rows_ * blocks_per_row_ * block_byte_size(format_);
        if (blocks_.size() != expected_bytes)
        {
            throw std::invalid_argument("quantized tensor block byte count does not match rows/cols/format");
        }
    }

    QuantFormat QuantizedTensor::format() const
    {
        return format_;
    }

    std::size_t QuantizedTensor::rows() const
    {
        return rows_;
    }

    std::size_t QuantizedTensor::cols() const
    {
        return cols_;
    }

    void QuantizedTensor::dequantize_row(std::size_t row, std::vector<float> &out) const
    {
        if (row >= rows_)
        {
            throw std::out_of_range("quantized tensor row index out of range");
        }

        out.resize(cols_);
        const std::size_t block_elems = block_size(format_);
        const std::size_t block_bytes = block_byte_size(format_);
        const std::size_t row_byte_offset = row * blocks_per_row_ * block_bytes;

        for (std::size_t b = 0; b < blocks_per_row_; ++b)
        {
            const std::byte *block = blocks_.data() + row_byte_offset + b * block_bytes;
            dequantize_block(format_, block, out.data() + b * block_elems);
        }
    }

    Tensor QuantizedTensor::dequantize() const
    {
        std::vector<float> values(rows_ * cols_);
        std::vector<float> row_buffer;
        for (std::size_t row = 0; row < rows_; ++row)
        {
            dequantize_row(row, row_buffer);
            std::copy(row_buffer.begin(), row_buffer.end(), values.begin() + row * cols_);
        }
        return Tensor({rows_, cols_}, std::move(values));
    }

} // namespace mini_inference::tensor
