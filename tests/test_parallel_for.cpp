#include "tensor/parallel_for.h"

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using mini_inference::tensor::parallel_for;

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

    // Runs parallel_for over [0, n) and asserts every index in that range was visited by
    // exactly one chunk (no gaps, no overlap), regardless of how many threads were used.
    void check_full_disjoint_coverage(std::size_t n, const std::string &label)
    {
        std::vector<char> touched(n, 0); // vector<char>, not vector<bool>: avoids bit-packed
                                          // storage, which would make disjoint writes unsafe.
        parallel_for(0, n, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i)
            {
                ++touched[i];
            }
        });

        for (std::size_t i = 0; i < n; ++i)
        {
            expect(touched[i] == 1, label + " index " + std::to_string(i) + " visited exactly once");
        }
    }

    void test_small_range_inline_fallback()
    {
        check_full_disjoint_coverage(10, "small range (inline fallback)");
    }

    void test_large_range_multithreaded()
    {
        check_full_disjoint_coverage(10000, "large range (multithreaded)");
    }

    void test_empty_range()
    {
        bool called = false;
        parallel_for(5, 5, [&](std::size_t, std::size_t) { called = true; });
        expect(!called, "empty range never invokes body");
    }

} // namespace

int main()
{
    test_small_range_inline_fallback();
    test_large_range_multithreaded();
    test_empty_range();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All parallel_for tests passed" << std::endl;
    return 0;
}
