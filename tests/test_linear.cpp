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

    // out_features well above parallel_for's threading threshold, so this exercises the
    // multithreaded path (not just the small-range inline fallback the test above takes).
    // All-ones weights and zero bias make every row's expected output simply sum(input).
    void test_large_out_features_uses_threaded_path()
    {
        constexpr std::size_t kInFeatures = 8;
        constexpr std::size_t kOutFeatures = 200;

        const std::vector<float> weights(kInFeatures * kOutFeatures, 1.0f);
        Linear layer(kInFeatures, kOutFeatures, weights, {});

        const std::vector<float> input_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        const float expected_sum = 36.0f; // 1+2+...+8

        Tensor input({1, kInFeatures}, input_values);
        Tensor output = layer.forward(input);

        expect(output.shape()[0] == 1, "large out_features batch size");
        expect(output.shape()[1] == kOutFeatures, "large out_features output size");
        for (std::size_t out_idx = 0; out_idx < kOutFeatures; ++out_idx)
        {
            expect_close(output.at({0, out_idx}), expected_sum,
                         "large out_features row " + std::to_string(out_idx));
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

    test_large_out_features_uses_threaded_path();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All linear layer tests passed" << std::endl;
    return 0;
}
