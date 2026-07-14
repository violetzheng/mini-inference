#include "layers/embedding.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using mini_inference::layers::Embedding;
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
    // Row i = {i, i}, so any lookup mistake (wrong row, off-by-one, transposed
    // indexing) produces an immediately obvious wrong value.
    Embedding embedding(4, 2, {0.0f, 0.0f,
                               1.0f, 1.0f,
                               2.0f, 2.0f,
                               3.0f, 3.0f});

    expect(embedding.vocab_size() == 4, "vocab_size");
    expect(embedding.hidden_dim() == 2, "hidden_dim");

    Tensor output = embedding.forward({2, 0, 2, 3});

    expect(output.rank() == 2, "output rank");
    expect(output.shape()[0] == 4, "output seq_len");
    expect(output.shape()[1] == 2, "output hidden_dim");
    expect_close(output.at({0, 0}), 2.0f, "row 0 (token 2) dim 0");
    expect_close(output.at({0, 1}), 2.0f, "row 0 (token 2) dim 1");
    expect_close(output.at({1, 0}), 0.0f, "row 1 (token 0) dim 0");
    expect_close(output.at({1, 1}), 0.0f, "row 1 (token 0) dim 1");
    expect_close(output.at({2, 0}), 2.0f, "row 2 (token 2, repeat) dim 0");
    expect_close(output.at({2, 1}), 2.0f, "row 2 (token 2, repeat) dim 1");
    expect_close(output.at({3, 0}), 3.0f, "row 3 (token 3) dim 0");
    expect_close(output.at({3, 1}), 3.0f, "row 3 (token 3) dim 1");

    Tensor empty_output = embedding.forward({});
    expect(empty_output.rank() == 2, "empty output rank");
    expect(empty_output.shape()[0] == 0, "empty output seq_len");
    expect(empty_output.shape()[1] == 2, "empty output hidden_dim");
    expect(empty_output.numel() == 0, "empty output numel");

    bool threw_zero_vocab = false;
    try
    {
        Embedding bad(0, 2);
        (void)bad;
    }
    catch (const std::invalid_argument &)
    {
        threw_zero_vocab = true;
    }
    expect(threw_zero_vocab, "zero vocab_size throws");

    bool threw_zero_hidden = false;
    try
    {
        Embedding bad(4, 0);
        (void)bad;
    }
    catch (const std::invalid_argument &)
    {
        threw_zero_hidden = true;
    }
    expect(threw_zero_hidden, "zero hidden_dim throws");

    bool threw_weight_mismatch = false;
    try
    {
        Embedding bad(4, 2, {0.0f, 0.0f, 1.0f});
        (void)bad;
    }
    catch (const std::invalid_argument &)
    {
        threw_weight_mismatch = true;
    }
    expect(threw_weight_mismatch, "weight count mismatch throws");

    bool threw_id_equal_vocab = false;
    try
    {
        (void)embedding.forward({4});
    }
    catch (const std::invalid_argument &)
    {
        threw_id_equal_vocab = true;
    }
    expect(threw_id_equal_vocab, "token id == vocab_size throws");

    bool threw_id_over_vocab = false;
    try
    {
        (void)embedding.forward({100});
    }
    catch (const std::invalid_argument &)
    {
        threw_id_over_vocab = true;
    }
    expect(threw_id_over_vocab, "token id > vocab_size throws");

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All embedding tests passed" << std::endl;
    return 0;
}
