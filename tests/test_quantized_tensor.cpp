#include "tensor/quantized_tensor.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using mini_inference::tensor::QuantFormat;
using mini_inference::tensor::QuantizedTensor;
using mini_inference::tensor::Tensor;

namespace
{

    int failures = 0;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            std::cerr << "FAILED: " << message << std::endl;
            ++failures;
        }
    }

    void expect_close(float actual, float expected, const std::string &message)
    {
        if (std::abs(actual - expected) > 1e-3f)
        {
            std::cerr << "FAILED: " << message << " (expected " << expected << ", got " << actual << ")" << std::endl;
            ++failures;
        }
    }

    std::vector<std::byte> make_bytes(std::initializer_list<int> values)
    {
        std::vector<std::byte> bytes;
        bytes.reserve(values.size());
        for (int v : values)
        {
            bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(v)));
        }
        return bytes;
    }

    // Two Q8_0 rows of 32 elements (one block each): row 0 has d=1.0, qs[0]=10 (rest 0);
    // row 1 has d=2.0, qs[0]=3, qs[1]=-3 (rest 0).
    std::vector<std::byte> two_row_q8_0_blocks()
    {
        std::vector<std::byte> row0 = make_bytes({0x00, 0x3C, 10});
        row0.resize(34, std::byte{0});

        std::vector<std::byte> row1 = make_bytes({0x00, 0x40, 3, 253 /* -3 as int8 */});
        row1.resize(34, std::byte{0});

        std::vector<std::byte> blocks = row0;
        blocks.insert(blocks.end(), row1.begin(), row1.end());
        return blocks;
    }

} // namespace

int main()
{
    QuantizedTensor qt(QuantFormat::Q8_0, 2, 32, two_row_q8_0_blocks());

    expect(qt.format() == QuantFormat::Q8_0, "format accessor");
    expect(qt.rows() == 2, "rows accessor");
    expect(qt.cols() == 32, "cols accessor");

    std::vector<float> row_buffer;
    qt.dequantize_row(0, row_buffer);
    expect(row_buffer.size() == 32, "dequantize_row 0 size");
    expect_close(row_buffer[0], 10.0f, "row 0 element 0");
    for (std::size_t i = 1; i < 32; ++i)
    {
        expect_close(row_buffer[i], 0.0f, "row 0 background element " + std::to_string(i));
    }

    qt.dequantize_row(1, row_buffer);
    expect_close(row_buffer[0], 6.0f, "row 1 element 0 (3 * d=2.0)");
    expect_close(row_buffer[1], -6.0f, "row 1 element 1 (-3 * d=2.0)");
    for (std::size_t i = 2; i < 32; ++i)
    {
        expect_close(row_buffer[i], 0.0f, "row 1 background element " + std::to_string(i));
    }

    const Tensor full = qt.dequantize();
    expect(full.shape()[0] == 2 && full.shape()[1] == 32, "dequantize() shape");
    expect_close(full.at({0, 0}), 10.0f, "dequantize() row 0 element 0");
    expect_close(full.at({1, 0}), 6.0f, "dequantize() row 1 element 0");
    expect_close(full.at({1, 1}), -6.0f, "dequantize() row 1 element 1");

    bool threw_row_out_of_range = false;
    try
    {
        qt.dequantize_row(2, row_buffer);
    }
    catch (const std::out_of_range &)
    {
        threw_row_out_of_range = true;
    }
    expect(threw_row_out_of_range, "dequantize_row past rows() throws out_of_range");

    bool threw_bad_col_count = false;
    try
    {
        QuantizedTensor bad(QuantFormat::Q8_0, 1, 31, std::vector<std::byte>(34, std::byte{0}));
        (void)bad;
    }
    catch (const std::invalid_argument &)
    {
        threw_bad_col_count = true;
    }
    expect(threw_bad_col_count, "cols not a multiple of block_size throws");

    bool threw_bad_byte_count = false;
    try
    {
        QuantizedTensor bad(QuantFormat::Q8_0, 1, 32, std::vector<std::byte>(10, std::byte{0}));
        (void)bad;
    }
    catch (const std::invalid_argument &)
    {
        threw_bad_byte_count = true;
    }
    expect(threw_bad_byte_count, "wrong block byte count throws");

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All quantized tensor tests passed" << std::endl;
    return 0;
}
