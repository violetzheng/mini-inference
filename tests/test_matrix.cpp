#include "tensor/matrix.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using mini_inference::tensor::matmul;
using mini_inference::tensor::matrix;

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
    matrix a(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    matrix b(3, 2, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    matrix result = matmul(a, b);
    expect(result.rows() == 2, "result row count");
    expect(result.cols() == 2, "result column count");
    expect_close(result.at(0, 0), 58.0f, "(0,0) entry");
    expect_close(result.at(0, 1), 64.0f, "(0,1) entry");
    expect_close(result.at(1, 0), 139.0f, "(1,0) entry");
    expect_close(result.at(1, 1), 154.0f, "(1,1) entry");

    matrix identity(2, 2, {1.0f, 0.0f, 0.0f, 1.0f});
    matrix identity_result = matmul(identity, a);
    expect(identity_result.rows() == 2, "identity rows");
    expect(identity_result.cols() == 3, "identity cols");
    expect_close(identity_result.at(0, 2), 3.0f, "identity preserves third column");

    bool threw = false;
    try
    {
        (void)matmul(a, matrix(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}));
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    expect(threw, "incompatible dimensions throw");

    matrix zero(1, 1, {0.0f});
    matrix zero_result = matmul(zero, zero);
    expect_close(zero_result.at(0, 0), 0.0f, "zero-sized-like multiply");

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All matrix tests passed" << std::endl;
    return 0;
}
