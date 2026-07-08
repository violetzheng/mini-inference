#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace mini_inference::tensor
{

    class Matrix
    {
    public:
        Matrix() = default;
        Matrix(std::size_t rows, std::size_t cols, std::vector<float> values = {});

        std::size_t rows() const;
        std::size_t cols() const;
        const std::vector<float> &values() const;

        float &at(std::size_t row, std::size_t col);
        float at(std::size_t row, std::size_t col) const;

    private:
        std::size_t rows_{0};
        std::size_t cols_{0};
        std::vector<float> values_{};
    };

    Matrix matmul(const Matrix &lhs, const Matrix &rhs);

} // namespace mini_inference::tensor
