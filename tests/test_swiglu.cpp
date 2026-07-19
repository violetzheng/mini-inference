#include "layers/swiglu.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using mini_inference::layers::GateActivation;
using mini_inference::layers::SwiGLU;
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

    float silu(float x)
    {
        return x / (1.0f + std::exp(-x));
    }

    // Matches fused_gelu_mul's tanh approximation (Gemma's "gelu_pytorch_tanh").
    float gelu_tanh(float x)
    {
        constexpr float kSqrt2OverPi = 0.7978845608028654f;
        return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
    }

} // namespace

int main()
{
    // hidden_dim=2, intermediate_dim=1 with sparse weights: gate isolates x0, up isolates x1,
    // so gate/up/fused/output are all hand-derivable from silu().
    const std::size_t hidden_dim = 2;
    const std::size_t intermediate_dim = 1;
    SwiGLU ffn(hidden_dim, intermediate_dim,
               {1.0f, 0.0f}, {},  // gate_weights (1x2): gate = x0
               {0.0f, 1.0f}, {},  // up_weights (1x2): up = x1
               {2.0f, -1.0f}, {}); // down_weights (2x1): out = fused * [2, -1]

    expect(ffn.hidden_dim() == 2, "hidden_dim");
    expect(ffn.intermediate_dim() == 1, "intermediate_dim");

    Tensor input({2, 2}, {1.0f, 2.0f,
                          -3.0f, 0.5f});
    Tensor output = ffn.forward(input);

    expect(output.rank() == 2, "output rank");
    expect(output.shape()[0] == 2, "output batch size");
    expect(output.shape()[1] == hidden_dim, "output feature count matches hidden_dim");

    const float fused_row0 = silu(1.0f) * 2.0f;
    expect_close(output.at({0, 0}), fused_row0 * 2.0f, "row 0 output dim 0");
    expect_close(output.at({0, 1}), fused_row0 * -1.0f, "row 0 output dim 1");

    const float fused_row1 = silu(-3.0f) * 0.5f;
    expect_close(output.at({1, 0}), fused_row1 * 2.0f, "row 1 output dim 0 (negative gate)");
    expect_close(output.at({1, 1}), fused_row1 * -1.0f, "row 1 output dim 1 (negative gate)");

    // Omitting bias args should behave the same as explicit zero bias (LLaMA-style, bias-free).
    bool threw_default_bias = false;
    try
    {
        SwiGLU default_bias_ffn(hidden_dim, intermediate_dim,
                                 {1.0f, 0.0f}, {},
                                 {0.0f, 1.0f}, {},
                                 {2.0f, -1.0f}, {});
        Tensor default_bias_output = default_bias_ffn.forward(input);
        expect_close(default_bias_output.at({0, 0}), fused_row0 * 2.0f, "default-bias row 0 output dim 0");
    }
    catch (const std::invalid_argument &)
    {
        threw_default_bias = true;
    }
    expect(!threw_default_bias, "constructing with omitted bias args does not throw");

    bool threw_rank = false;
    try
    {
        (void)ffn.forward(Tensor({2}, {1.0f, 2.0f}));
    }
    catch (const std::invalid_argument &)
    {
        threw_rank = true;
    }
    expect(threw_rank, "rank mismatch throws");

    bool threw_dim_mismatch = false;
    try
    {
        (void)ffn.forward(Tensor({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    }
    catch (const std::invalid_argument &)
    {
        threw_dim_mismatch = true;
    }
    expect(threw_dim_mismatch, "hidden_dim mismatch throws");

    bool threw_zero_hidden = false;
    try
    {
        SwiGLU zero_hidden(0, intermediate_dim);
        (void)zero_hidden;
    }
    catch (const std::invalid_argument &)
    {
        threw_zero_hidden = true;
    }
    expect(threw_zero_hidden, "zero hidden_dim throws");

    bool threw_zero_intermediate = false;
    try
    {
        SwiGLU zero_intermediate(hidden_dim, 0);
        (void)zero_intermediate;
    }
    catch (const std::invalid_argument &)
    {
        threw_zero_intermediate = true;
    }
    expect(threw_zero_intermediate, "zero intermediate_dim throws");

    bool threw_gate_weights = false;
    try
    {
        SwiGLU bad_gate(hidden_dim, intermediate_dim, {1.0f, 0.0f, 0.0f});
        (void)bad_gate;
    }
    catch (const std::invalid_argument &)
    {
        threw_gate_weights = true;
    }
    expect(threw_gate_weights, "gate_weights size mismatch throws");

    bool threw_gate_bias = false;
    try
    {
        SwiGLU bad_gate_bias(hidden_dim, intermediate_dim, {1.0f, 0.0f}, {0.0f, 0.0f});
        (void)bad_gate_bias;
    }
    catch (const std::invalid_argument &)
    {
        threw_gate_bias = true;
    }
    expect(threw_gate_bias, "gate_bias size mismatch throws");

    bool threw_up_weights = false;
    try
    {
        SwiGLU bad_up(hidden_dim, intermediate_dim, {1.0f, 0.0f}, {}, {0.0f, 1.0f, 1.0f});
        (void)bad_up;
    }
    catch (const std::invalid_argument &)
    {
        threw_up_weights = true;
    }
    expect(threw_up_weights, "up_weights size mismatch throws");

    bool threw_down_weights = false;
    try
    {
        // down_weights must be hidden_dim * intermediate_dim = 2, not 3.
        SwiGLU bad_down(hidden_dim, intermediate_dim, {1.0f, 0.0f}, {}, {0.0f, 1.0f}, {}, {2.0f, -1.0f, 0.0f});
        (void)bad_down;
    }
    catch (const std::invalid_argument &)
    {
        threw_down_weights = true;
    }
    expect(threw_down_weights, "down_weights size mismatch throws");

    // GateActivation::kGelu (GeGLU): same wiring as above, gated with gelu_tanh instead.
    {
        SwiGLU gelu_ffn(hidden_dim, intermediate_dim,
                         {1.0f, 0.0f}, {},
                         {0.0f, 1.0f}, {},
                         {2.0f, -1.0f}, {},
                         GateActivation::kGelu);

        Tensor gelu_output = gelu_ffn.forward(input);

        const float gelu_fused_row0 = gelu_tanh(1.0f) * 2.0f;
        expect_close(gelu_output.at({0, 0}), gelu_fused_row0 * 2.0f, "gelu row 0 output dim 0");
        expect_close(gelu_output.at({0, 1}), gelu_fused_row0 * -1.0f, "gelu row 0 output dim 1");

        const float gelu_fused_row1 = gelu_tanh(-3.0f) * 0.5f;
        expect_close(gelu_output.at({1, 0}), gelu_fused_row1 * 2.0f, "gelu row 1 output dim 0 (negative gate)");
        expect_close(gelu_output.at({1, 1}), gelu_fused_row1 * -1.0f, "gelu row 1 output dim 1 (negative gate)");
    }

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All swiglu tests passed" << std::endl;
    return 0;
}
