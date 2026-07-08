#include "tensor/matrix.h"

#include <cassert>

namespace mini_inference::tensor
{

    matrix::matrix(std::size_t rows, std::size_t cols, std::vector<float> values)
        : rows_(rows), cols_(cols)
    {
        if (values.empty())
        {
            values_.assign(rows_ * cols_, 0.0f);
        }
        else
        {
            assert(values.size() == rows_ * cols_);
            values_ = std::move(values);
        }
    }

    std::size_t matrix::rows() const
    {
        return rows_;
    }

    std::size_t matrix::cols() const
    {
        return cols_;
    }

    const std::vector<float> &matrix::values() const
    {
        return values_;
    }

    float &matrix::at(std::size_t row, std::size_t col)
    {
        assert(row < rows_ && col < cols_);
        return values_[row * cols_ + col];
    }

    float matrix::at(std::size_t row, std::size_t col) const
    {
        assert(row < rows_ && col < cols_);
        return values_[row * cols_ + col];
    }

    matrix matmul(const matrix &lhs, const matrix &rhs)
    {
        assert(lhs.cols() > 0 && lhs.rows() > 0);
        assert(rhs.cols() > 0 && rhs.rows() > 0);

        if (lhs.cols() != rhs.rows())
        {
            throw std::invalid_argument("matrix dimensions are incompatible for multiplication");
        }

        matrix result(lhs.rows(), rhs.cols());
        for (std::size_t row = 0; row < lhs.rows(); ++row)
        {
            for (std::size_t col = 0; col < rhs.cols(); ++col)
            {
                float sum = 0.0f;
                for (std::size_t inner = 0; inner < lhs.cols(); ++inner)
                {
                    sum += lhs.at(row, inner) * rhs.at(inner, col);
                }
                result.at(row, col) = sum;
            }
        }

        return result;
    }

} // namespace mini_inference::tensor
