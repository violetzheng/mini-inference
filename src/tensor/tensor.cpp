#include "tensor/tensor.h"

#include <cassert>
#include <numeric>

namespace mini_inference::tensor
{

    namespace
    {

        std::size_t compute_numel(const std::vector<std::size_t> &shape)
        {
            return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>());
        }

    } // namespace

    Tensor::Tensor(std::vector<std::size_t> shape, std::vector<float> values)
        : shape_(std::move(shape)), values_(std::move(values))
    {
        if (shape_.empty())
        {
            throw std::invalid_argument("tensor shape must not be empty");
        }

        if (values_.empty())
        {
            values_.assign(compute_numel(shape_), 0.0f);
        }
        else if (values_.size() != compute_numel(shape_))
        {
            throw std::invalid_argument("tensor value count does not match shape");
        }
    }

    const std::vector<std::size_t> &Tensor::shape() const
    {
        return shape_;
    }

    std::size_t Tensor::rank() const
    {
        return shape_.size();
    }

    std::size_t Tensor::numel() const
    {
        return values_.size();
    }

    const std::vector<float> &Tensor::values() const
    {
        return values_;
    }

    float &Tensor::at(const std::vector<std::size_t> &indices)
    {
        if (indices.size() != shape_.size())
        {
            throw std::invalid_argument("tensor index rank does not match shape");
        }

        std::size_t flat_index = 0;
        std::size_t stride = 1;
        for (std::size_t i = shape_.size(); i-- > 0;)
        {
            const std::size_t dim = shape_[i];
            const std::size_t idx = indices[i];
            if (idx >= dim)
            {
                throw std::out_of_range("tensor index out of bounds");
            }
            flat_index += idx * stride;
            stride *= dim;
        }

        return values_[flat_index];
    }

    float Tensor::at(const std::vector<std::size_t> &indices) const
    {
        return const_cast<Tensor *>(this)->at(indices);
    }

    float &Tensor::at(std::size_t flat_index)
    {
        if (flat_index >= values_.size())
        {
            throw std::out_of_range("tensor flat index out of bounds");
        }
        return values_[flat_index];
    }

    float Tensor::at(std::size_t flat_index) const
    {
        if (flat_index >= values_.size())
        {
            throw std::out_of_range("tensor flat index out of bounds");
        }
        return values_[flat_index];
    }

    Tensor Tensor::reshape(const std::vector<std::size_t> &new_shape) const
    {
        const std::size_t new_numel = compute_numel(new_shape);
        if (new_numel != values_.size())
        {
            throw std::invalid_argument("tensor reshape size mismatch");
        }
        return Tensor(new_shape, values_);
    }

} // namespace mini_inference::tensor
