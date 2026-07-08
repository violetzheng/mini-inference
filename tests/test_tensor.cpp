#include "tensor/tensor.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using mini_inference::tensor::tensor;

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
    tensor values({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    expect(values.rank() == 2, "tensor rank");
    expect(values.shape().size() == 2, "tensor shape size");
    expect(values.shape()[0] == 2, "tensor first dimension");
    expect(values.shape()[1] == 3, "tensor second dimension");
    expect_close(values.at({1, 2}), 6.0f, "2D access");
    expect_close(values.at(5), 6.0f, "flat access");

    tensor reshaped = values.reshape({3, 2});
    expect(reshaped.shape()[0] == 3, "reshape first dimension");
    expect(reshaped.shape()[1] == 2, "reshape second dimension");
    expect_close(reshaped.at({2, 1}), 6.0f, "reshape preserves values");

    bool threw = false;
    try
    {
        (void)tensor({2, 2}, {1.0f, 2.0f, 3.0f});
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    expect(threw, "constructor validates element count");

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All tensor tests passed" << std::endl;
    return 0;
}
