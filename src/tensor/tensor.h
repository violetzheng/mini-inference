#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_inference::tensor
{

    class tensor
    {
    public:
        tensor() = default;
        tensor(std::vector<std::size_t> shape, std::vector<float> values = {});

        const std::vector<std::size_t> &shape() const;
        std::size_t rank() const;
        std::size_t numel() const;
        const std::vector<float> &values() const;

        float &at(const std::vector<std::size_t> &indices);
        float at(const std::vector<std::size_t> &indices) const;

        float &at(std::size_t flat_index);
        float at(std::size_t flat_index) const;

        tensor reshape(const std::vector<std::size_t> &new_shape) const;

    private:
        std::vector<std::size_t> shape_{};
        std::vector<float> values_{};
    };

} // namespace mini_inference::tensor
