#include "layers/linear.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using mini_inference::layers::Linear;
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
        if (std::abs(actual - expected) > 1e-5f)
        {
            std::cerr << "FAILED: " << message << " (expected " << expected << ", got " << actual << ")" << std::endl;
            ++failures;
        }
    }

} // namespace

int main()
{
    Linear layer(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, {0.5f, -0.5f});

    Tensor input({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor output = layer.forward(input);

    expect(output.rank() == 2, "output rank");
    expect(output.shape()[0] == 2, "batch size");
    expect(output.shape()[1] == 2, "output size");
    expect_close(output.at({0, 0}), 1.0f * 1.0f + 2.0f * 2.0f + 0.5f, "first output");
    expect_close(output.at({0, 1}), 1.0f * 3.0f + 2.0f * 4.0f - 0.5f, "second output");
    expect_close(output.at({1, 0}), 3.0f * 1.0f + 4.0f * 2.0f + 0.5f, "third output");
    expect_close(output.at({1, 1}), 3.0f * 3.0f + 4.0f * 4.0f - 0.5f, "fourth output");

    bool threw = false;
    try
    {
        (void)layer.forward(Tensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    expect(threw, "shape mismatch throws");

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All linear layer tests passed" << std::endl;
    return 0;
}
