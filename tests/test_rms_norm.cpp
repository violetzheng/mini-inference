#include "layers/rms_norm.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using mini_inference::layers::RmsNorm;
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
    RmsNorm layer(2);
    Tensor input({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor output = layer.forward(input);

    expect(output.rank() == 2, "output rank");
    expect(output.shape()[0] == 2, "batch size");
    expect(output.shape()[1] == 2, "output size");

    const float expected_scale = std::sqrt((1.0f * 1.0f + 2.0f * 2.0f) / 2.0f + 1e-5f);
    expect_close(output.at({0, 0}), (1.0f / expected_scale) * 1.0f, "first normalized value");
    expect_close(output.at({0, 1}), (2.0f / expected_scale) * 1.0f, "second normalized value");

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

    std::cout << "All RMSNorm tests passed" << std::endl;
    return 0;
}
