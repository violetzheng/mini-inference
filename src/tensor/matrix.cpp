#include "tensor/matrix.h"

#include <cassert>

namespace mini_inference::tensor
{

    Matrix::Matrix(std::size_t rows, std::size_t cols, std::vector<float> values)
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

    std::size_t Matrix::rows() const
    {
        return rows_;
    }

    std::size_t Matrix::cols() const
    {
        return cols_;
    }

    const std::vector<float> &Matrix::values() const
    {
        return values_;
    }

    float &Matrix::at(std::size_t row, std::size_t col)
    {
        assert(row < rows_ && col < cols_);
        return values_[row * cols_ + col];
    }

    float Matrix::at(std::size_t row, std::size_t col) const
    {
        assert(row < rows_ && col < cols_);
        return values_[row * cols_ + col];
    }

    Matrix matmul(const Matrix &lhs, const Matrix &rhs)
    {
        assert(lhs.cols() > 0 && lhs.rows() > 0);
        assert(rhs.cols() > 0 && rhs.rows() > 0);

        if (lhs.cols() != rhs.rows())
        {
            throw std::invalid_argument("matrix dimensions are incompatible for multiplication");
        }

        Matrix result(lhs.rows(), rhs.cols());
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
