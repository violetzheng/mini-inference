#include "tensor/quant_blocks.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using mini_inference::tensor::block_byte_size;
using mini_inference::tensor::block_size;
using mini_inference::tensor::dequantize_block_q4_0;
using mini_inference::tensor::dequantize_block_q4_k;
using mini_inference::tensor::dequantize_block_q8_0;
using mini_inference::tensor::QuantFormat;

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
        if (std::abs(actual - expected) > 1e-3f)
        {
            std::cerr << "FAILED: " << message << " (expected " << expected << ", got " << actual << ")" << std::endl;
            ++failures;
        }
    }

    std::vector<std::byte> make_bytes(std::initializer_list<int> values)
    {
        std::vector<std::byte> bytes;
        bytes.reserve(values.size());
        for (int v : values)
        {
            bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(v)));
        }
        return bytes;
    }

    void test_sizes()
    {
        expect(block_size(QuantFormat::Q8_0) == 32, "Q8_0 block_size");
        expect(block_byte_size(QuantFormat::Q8_0) == 34, "Q8_0 block_byte_size");
        expect(block_size(QuantFormat::Q4_0) == 32, "Q4_0 block_size");
        expect(block_byte_size(QuantFormat::Q4_0) == 18, "Q4_0 block_byte_size");
        expect(block_size(QuantFormat::Q4_K) == 256, "Q4_K block_size");
        expect(block_byte_size(QuantFormat::Q4_K) == 144, "Q4_K block_byte_size");
    }

    void test_q8_0()
    {
        // d = 2.0 (fp16 0x4000, little-endian bytes 0x00, 0x40), qs = {1, -1, 127, -128, 0...0}
        std::vector<std::byte> block = make_bytes({0x00, 0x40,
                                                     1, 255 /*-1*/, 127, 128 /*-128*/});
        block.resize(34, std::byte{0});

        std::vector<float> out(32, 0.0f);
        dequantize_block_q8_0(block.data(), out.data());

        expect_close(out[0], 2.0f, "q8_0 element 0");
        expect_close(out[1], -2.0f, "q8_0 element 1");
        expect_close(out[2], 254.0f, "q8_0 element 2");
        expect_close(out[3], -256.0f, "q8_0 element 3");
        for (std::size_t i = 4; i < 32; ++i)
        {
            expect_close(out[i], 0.0f, "q8_0 background element " + std::to_string(i));
        }
    }

    void test_q4_0()
    {
        // d = 1.0 (fp16 0x3C00). qs[0] = 0x9F: low nibble 0xF=15 -> element 0, high nibble
        // 0x9=9 -> element 16. Remaining qs bytes = 0x88 (low/high nibble 8 -> value 0).
        std::vector<std::byte> block = make_bytes({0x00, 0x3C, 0x9F});
        for (int i = 0; i < 15; ++i)
        {
            block.push_back(static_cast<std::byte>(0x88));
        }

        std::vector<float> out(32, 0.0f);
        dequantize_block_q4_0(block.data(), out.data());

        expect_close(out[0], 7.0f, "q4_0 element 0 (low nibble 15 - 8)");
        expect_close(out[16], 1.0f, "q4_0 element 16 (high nibble 9 - 8)");
        for (std::size_t i = 1; i < 16; ++i)
        {
            expect_close(out[i], 0.0f, "q4_0 background low-nibble element " + std::to_string(i));
            expect_close(out[i + 16], 0.0f, "q4_0 background high-nibble element " + std::to_string(i + 16));
        }
    }

    void test_q4_k()
    {
        // d = dmin = 1.0 (fp16 0x3C00), so per-subblock value = sc_j * q - m_j directly.
        // scales[12] chosen to exercise both get_scale_min_k4 branches (j<4 and j>=4):
        //   j=0..3 read directly from scales[0..3]/scales[4..7]:
        //     (sc,m) = (5,1), (6,2), (7,3), (8,4)
        //   j=4..7 combine scales[8..11]'s nibbles (top bits of scales[0..7] are all 0 here):
        //     (sc,m) = (3,9), (5,10), (7,11), (2,12)
        std::vector<std::byte> block = make_bytes({0x00, 0x3C, 0x00, 0x3C,
                                                     5, 6, 7, 8, 1, 2, 3, 4, 0x93, 0xA5, 0xB7, 0xC2});
        std::vector<std::byte> qs(128, std::byte{0});
        qs[0] = static_cast<std::byte>(0xA3);  // low nibble 3, high nibble 10 (0xA)
        qs[32] = static_cast<std::byte>(0xC4); // low nibble 4, high nibble 12 (0xC)
        qs[64] = static_cast<std::byte>(0x5D); // low nibble 13 (0xD), high nibble 5
        qs[96] = static_cast<std::byte>(0xE7); // low nibble 7, high nibble 14 (0xE)
        block.insert(block.end(), qs.begin(), qs.end());

        expect(block.size() == 144, "q4_k test block byte count");

        std::vector<float> out(256, 0.0f);
        dequantize_block_q4_k(block.data(), out.data());

        // chunk 0: j=0 (sc=5,m=1) for out[0..31] low nibbles of qs[0..31];
        //          j=1 (sc=6,m=2) for out[32..63] high nibbles of qs[0..31].
        expect_close(out[0], 5.0f * 3 - 1, "q4_k element 0 (sub-block j=0)");
        expect_close(out[1], 5.0f * 0 - 1, "q4_k background element 1 (sub-block j=0, q=0)");
        expect_close(out[32], 6.0f * 10 - 2, "q4_k element 32 (sub-block j=1)");
        expect_close(out[33], 6.0f * 0 - 2, "q4_k background element 33 (sub-block j=1, q=0)");

        // chunk 1: j=2 (sc=7,m=3) for out[64..95]; j=3 (sc=8,m=4) for out[96..127].
        expect_close(out[64], 7.0f * 4 - 3, "q4_k element 64 (sub-block j=2)");
        expect_close(out[96], 8.0f * 12 - 4, "q4_k element 96 (sub-block j=3)");

        // chunk 2: j=4 (sc=3,m=9) for out[128..159]; j=5 (sc=5,m=10) for out[160..191].
        expect_close(out[128], 3.0f * 13 - 9, "q4_k element 128 (sub-block j=4)");
        expect_close(out[160], 5.0f * 5 - 10, "q4_k element 160 (sub-block j=5)");

        // chunk 3: j=6 (sc=7,m=11) for out[192..223]; j=7 (sc=2,m=12) for out[224..255].
        expect_close(out[192], 7.0f * 7 - 11, "q4_k element 192 (sub-block j=6)");
        expect_close(out[224], 2.0f * 14 - 12, "q4_k element 224 (sub-block j=7)");
    }

} // namespace

int main()
{
    test_sizes();
    test_q8_0();
    test_q4_0();
    test_q4_k();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All quant block tests passed" << std::endl;
    return 0;
}
