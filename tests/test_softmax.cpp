#include "layers/softmax.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using mini_inference::layers::Softmax;
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
    Softmax layer(1);
    Tensor input({2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f});
    Tensor output = layer.forward(input);

    expect(output.rank() == 2, "output rank");
    expect(output.shape()[0] == 2, "batch size");
    expect(output.shape()[1] == 3, "output size");

    const float sum0 = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
    const float sum1 = 3.0f * std::exp(1.0f);
    expect_close(output.at({0, 0}), std::exp(1.0f) / sum0, "first softmax value");
    expect_close(output.at({0, 2}), std::exp(3.0f) / sum0, "third softmax value");
    expect_close(output.at({1, 0}), std::exp(1.0f) / sum1, "second row first value");

    bool threw = false;
    try
    {
        (void)layer.forward(Tensor({3}, {1.0f, 2.0f, 3.0f}));
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

    std::cout << "All softmax tests passed" << std::endl;
    return 0;
}
