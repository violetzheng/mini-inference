#include "layers/rope.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using mini_inference::layers::RoPE;
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
        if (std::abs(actual - expected) > 1e-4f)
        {
            std::cerr << "FAILED: " << message << " (expected " << expected << ", got " << actual << ")" << std::endl;
            ++failures;
        }
    }

} // namespace

int main()
{
    RoPE rope(4, 10000.0f, 8);

    // Position 0 has a zero rotation angle, so the first row must be unchanged.
    Tensor input({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f,
                          1.0f, 2.0f, 3.0f, 4.0f});
    Tensor output = rope.forward(input);

    expect(output.rank() == 2, "output rank");
    expect(output.shape()[0] == 2, "sequence length");
    expect(output.shape()[1] == 4, "feature dim");

    expect_close(output.at({0, 0}), 1.0f, "position 0 pair 0 even (identity rotation)");
    expect_close(output.at({0, 1}), 2.0f, "position 0 pair 0 odd (identity rotation)");
    expect_close(output.at({0, 2}), 3.0f, "position 0 pair 1 even (identity rotation)");
    expect_close(output.at({0, 3}), 4.0f, "position 0 pair 1 odd (identity rotation)");

    const float inv_freq0 = 1.0f;
    const float inv_freq1 = 1.0f / std::pow(10000.0f, 2.0f / 4.0f);
    const float angle0 = inv_freq0;
    const float angle1 = inv_freq1;

    expect_close(output.at({1, 0}), 1.0f * std::cos(angle0) - 2.0f * std::sin(angle0), "position 1 pair 0 even");
    expect_close(output.at({1, 1}), 1.0f * std::sin(angle0) + 2.0f * std::cos(angle0), "position 1 pair 0 odd");
    expect_close(output.at({1, 2}), 3.0f * std::cos(angle1) - 4.0f * std::sin(angle1), "position 1 pair 1 even");
    expect_close(output.at({1, 3}), 3.0f * std::sin(angle1) + 4.0f * std::cos(angle1), "position 1 pair 1 odd");

    // An explicit position id must match the equivalent contiguous offset.
    Tensor single_row({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor via_offset = rope.forward(single_row, 1);
    Tensor via_positions = rope.forward(single_row, std::vector<std::size_t>{1});
    for (std::size_t i = 0; i < 4; ++i)
    {
        expect_close(via_offset.at({0, i}), via_positions.at({0, i}), "offset and explicit position agree");
    }

    bool threw_odd_dim = false;
    try
    {
        RoPE bad_rope(3);
        (void)bad_rope;
    }
    catch (const std::invalid_argument &)
    {
        threw_odd_dim = true;
    }
    expect(threw_odd_dim, "odd dimension throws");

    bool threw_rank = false;
    try
    {
        (void)rope.forward(Tensor({4}, {1.0f, 2.0f, 3.0f, 4.0f}));
    }
    catch (const std::invalid_argument &)
    {
        threw_rank = true;
    }
    expect(threw_rank, "rank mismatch throws");

    bool threw_dim_mismatch = false;
    try
    {
        (void)rope.forward(Tensor({1, 2}, {1.0f, 2.0f}));
    }
    catch (const std::invalid_argument &)
    {
        threw_dim_mismatch = true;
    }
    expect(threw_dim_mismatch, "feature dim mismatch throws");

    bool threw_position_overflow = false;
    try
    {
        (void)rope.forward(single_row, 100);
    }
    catch (const std::invalid_argument &)
    {
        threw_position_overflow = true;
    }
    expect(threw_position_overflow, "position beyond max_position_embeddings throws");

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All RoPE tests passed" << std::endl;
    return 0;
}
