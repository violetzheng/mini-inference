#include "tensor/quant_blocks.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using mini_inference::tensor::block_byte_size;
using mini_inference::tensor::block_size;
using mini_inference::tensor::dequantize_block_q2_k;
using mini_inference::tensor::dequantize_block_q3_k;
using mini_inference::tensor::dequantize_block_q4_0;
using mini_inference::tensor::dequantize_block_q4_k;
using mini_inference::tensor::dequantize_block_q5_k;
using mini_inference::tensor::dequantize_block_q6_k;
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
        expect(block_size(QuantFormat::Q6_K) == 256, "Q6_K block_size");
        expect(block_byte_size(QuantFormat::Q6_K) == 210, "Q6_K block_byte_size");
        expect(block_size(QuantFormat::Q5_K) == 256, "Q5_K block_size");
        expect(block_byte_size(QuantFormat::Q5_K) == 176, "Q5_K block_byte_size");
        expect(block_size(QuantFormat::Q2_K) == 256, "Q2_K block_size");
        expect(block_byte_size(QuantFormat::Q2_K) == 84, "Q2_K block_byte_size");
        expect(block_size(QuantFormat::Q3_K) == 256, "Q3_K block_size");
        expect(block_byte_size(QuantFormat::Q3_K) == 110, "Q3_K block_byte_size");
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

    void test_q6_k()
    {
        // d = 1.0 (fp16 0x3C00, bytes at the block's end: offset 208-209). Only
        // scales[0,2,4,6] (half 0, is=0 group) and scales[8] (half 1, is=0 group) are
        // non-zero; every scale in the is=1 group (scales[1,3,5,7,9,...]) is left at its
        // zero default, so output positions from that group are 0 regardless of quant
        // bits - see the background-element checks below.
        //
        // Half 0, l=0: ql[0]=0xD8, ql[32]=0x23, qh[0]=0xEA reassemble to raw 6-bit
        // quants 40, 35, 45, 50 for q1..q4 (see dequantize_block_q6_k's bit layout),
        // i.e. q1=8, q2=3, q3=13, q4=18 after the -32 re-center.
        std::vector<std::byte> block(210, std::byte{0});
        block[0] = static_cast<std::byte>(0xD8);
        block[32] = static_cast<std::byte>(0x23);
        block[128] = static_cast<std::byte>(0xEA);
        block[192 + 0] = static_cast<std::byte>(5);
        block[192 + 2] = static_cast<std::byte>(6);
        block[192 + 4] = static_cast<std::byte>(7);
        block[192 + 6] = static_cast<std::byte>(8);
        // Half 1, l=0: ql[64]=0x0A, qh[32] (global byte 160)=0x02 -> raw quant 42, q1=10.
        block[64] = static_cast<std::byte>(0x0A);
        block[160] = static_cast<std::byte>(0x02);
        block[192 + 8] = static_cast<std::byte>(2);
        block[208] = static_cast<std::byte>(0x00);
        block[209] = static_cast<std::byte>(0x3C);

        std::vector<float> out(256, 0.0f);
        dequantize_block_q6_k(block.data(), out.data());

        expect_close(out[0], 40.0f, "q6_k half0 element 0 (d=1, sc=5, q1=8)");
        expect_close(out[32], 18.0f, "q6_k half0 element 32 (d=1, sc=6, q2=3)");
        expect_close(out[64], 91.0f, "q6_k half0 element 64 (d=1, sc=7, q3=13)");
        expect_close(out[96], 144.0f, "q6_k half0 element 96 (d=1, sc=8, q4=18)");
        expect_close(out[128], 20.0f, "q6_k half1 element 128 (d=1, sc=2, q1=10)");

        // Elements with l in [16,32) (is=1) use scales[1,3,5,7,9,...], which were all
        // left at their zero default, so those positions are 0 regardless of quant bits
        // - unlike e.g. out[1], which shares out[0]'s is=0/scales[0] group and so is NOT
        // zero (it picks up q1's background quant bits scaled by the non-zero scales[0]).
        expect_close(out[16], 0.0f, "q6_k background element 16 (is=1, zero scale)");
        expect_close(out[48], 0.0f, "q6_k background element 48 (is=1, zero scale)");
        expect_close(out[80], 0.0f, "q6_k background element 80 (is=1, zero scale)");
        expect_close(out[144], 0.0f, "q6_k background element 144 (half1, is=1, zero scale)");
        expect_close(out[255], 0.0f, "q6_k background element 255 (half1, is=1, zero scale)");
    }

    void test_q5_k()
    {
        // d = dmin = 1.0. Sub-block scale/min packed exactly like Q4_K (get_scale_min_k4):
        // scales[0]=5 (chunk0's sc0), scales[4]=2 (chunk0's m0); scales[3]=6, scales[7]=1
        // (chunk1's sc1/m1, via is = chunk*2+1 = 3). Chunk0/1's OTHER sub-block (sc1/m0
        // resp.) are left at zero, making that half of each chunk's output exactly 0.
        std::vector<std::byte> block(176, std::byte{0});
        block[0] = static_cast<std::byte>(0x00);
        block[1] = static_cast<std::byte>(0x3C); // d = 1.0
        block[2] = static_cast<std::byte>(0x00);
        block[3] = static_cast<std::byte>(0x3C); // dmin = 1.0
        block[4 + 0] = static_cast<std::byte>(5); // scales[0] (chunk0 lo sc)
        block[4 + 3] = static_cast<std::byte>(6);  // scales[3] (chunk1 hi sc)
        block[4 + 4] = static_cast<std::byte>(2);  // scales[4] (chunk0 lo min)
        block[4 + 7] = static_cast<std::byte>(1);  // scales[7] (chunk1 hi min)
        block[16 + 0] = static_cast<std::byte>(0x01);          // qh[0]: bit0 set
        block[4 + 12 + 32 + 0] = static_cast<std::byte>(0x08); // qs[0]: low nibble 8
        block[4 + 12 + 32 + 32] = static_cast<std::byte>(0x05); // qs[32]: low nibble 5

        std::vector<float> out(256, 0.0f);
        dequantize_block_q5_k(block.data(), out.data());

        // Chunk0 lo, l=0: q = (qs[0]&0xF) | (qh[0]&u1(=1) ? 16:0) = 8|16 = 24.
        // v = d*sc*q - dmin*m = 1*5*24 - 1*2 = 118.
        expect_close(out[0], 118.0f, "q5_k chunk0 lo element 0 (5th bit set, q=24)");
        // Same chunk0-lo scale/min group, but background quant bits (q=0): v = 5*0-2 = -2.
        expect_close(out[1], -2.0f, "q5_k chunk0 lo background element 1");
        // Chunk0 hi sub-block's scale/min (scales[1]/scales[5]) are both left at zero.
        expect_close(out[32], 0.0f, "q5_k chunk0 hi element 32 (zero scale/min)");
        // Chunk1 lo sub-block's scale/min (scales[2]/scales[6]) are both left at zero.
        expect_close(out[64], 0.0f, "q5_k chunk1 lo element 64 (zero scale/min)");
        // Chunk1 hi, l=0: u2 = 2<<2 = 8; qh[0]=0x01, bit3 unset -> +0. q = qs[32]>>4 = 0.
        // v = d*6*0 - 1*1 = -1.
        expect_close(out[96], -1.0f, "q5_k chunk1 hi element 96 (5th bit unset, q=0)");
    }

    void test_q2_k()
    {
        // d = dmin = 1.0. scales[0] = 0x35 (min=3, scale=5) for chunk0/j=0's lo
        // sub-block; scales[1] = 0x12 (min=1, scale=2) for its hi sub-block; scales[8]
        // = 0x24 (min=2, scale=4) for chunk1/j=0's lo sub-block. scales[2]/[3] (chunk0
        // j=1's lo/hi) are left at zero, so that sub-block's output is exactly 0.
        std::vector<std::byte> block(84, std::byte{0});
        block[0] = static_cast<std::byte>(0x35); // scales[0]
        block[1] = static_cast<std::byte>(0x12); // scales[1]
        block[8] = static_cast<std::byte>(0x24); // scales[8]
        block[16 + 0] = static_cast<std::byte>(0x03);  // qs[0]: low 2 bits = 3
        block[16 + 16] = static_cast<std::byte>(0x03); // qs[16]: low 2 bits = 3
        block[16 + 32] = static_cast<std::byte>(0x01); // qs[32]: low 2 bits = 1
        block[80] = static_cast<std::byte>(0x00);
        block[81] = static_cast<std::byte>(0x3C); // d = 1.0
        block[82] = static_cast<std::byte>(0x00);
        block[83] = static_cast<std::byte>(0x3C); // dmin = 1.0

        std::vector<float> out(256, 0.0f);
        dequantize_block_q2_k(block.data(), out.data());

        // Chunk0/j=0 lo, l=0: v = d*5*3 - dmin*3 = 15-3 = 12.
        expect_close(out[0], 12.0f, "q2_k chunk0 j0 lo element 0");
        // Chunk0/j=0 hi, l=0: v = d*2*3 - dmin*1 = 6-1 = 5.
        expect_close(out[16], 5.0f, "q2_k chunk0 j0 hi element 16");
        // Chunk0/j=1 lo/hi sub-blocks (scales[2]/[3]) are both zero.
        expect_close(out[32], 0.0f, "q2_k chunk0 j1 lo element 32 (zero scale/min)");
        expect_close(out[48], 0.0f, "q2_k chunk0 j1 hi element 48 (zero scale/min)");
        // Chunk1/j=0 lo, l=0: v = d*4*1 - dmin*2 = 4-2 = 2.
        expect_close(out[128], 2.0f, "q2_k chunk1 j0 lo element 128");
    }

    void test_q3_k()
    {
        // d = 1.0. scales_raw[0] = 0x0D, scales_raw[8] = 0x02 unpack (via
        // unpack_q3_k_scales) to scales_unpacked[0] = 45 (sc = 45-32 = 13, used for
        // chunk0/j=0's lo sub-block) and scales_unpacked[8] = 0 (sc = 0-32 = -32, used
        // for chunk1/j=0's lo sub-block, since `is` keeps counting across chunks).
        std::vector<std::byte> block(110, std::byte{0});
        block[32 + 0] = static_cast<std::byte>(0x03); // qs[0]: low 2 bits = 3
        block[96 + 0] = static_cast<std::byte>(0x0D); // scales_raw[0]
        block[96 + 8] = static_cast<std::byte>(0x02); // scales_raw[8]
        block[108] = static_cast<std::byte>(0x00);
        block[109] = static_cast<std::byte>(0x3C); // d = 1.0
        // hmask left all-zero: every hmask bit tested below is "unset" -> subtract 4.

        std::vector<float> out(256, 0.0f);
        dequantize_block_q3_k(block.data(), out.data());

        // Chunk0/j=0 lo, l=0: sc=13, m=1 (initial rotating mask), hmask[0] bit0 unset.
        // bits = qs[0]&3 = 3; centered = 3 - 4 = -1. v = d*13*(-1) = -13.
        expect_close(out[0], -13.0f, "q3_k chunk0 lo element 0 (hot quant, hmask unset)");
        // Same sub-block, background quant bits (q=0): centered = 0-4 = -4. v=13*-4=-52.
        expect_close(out[1], -52.0f, "q3_k chunk0 lo background element 1");
        // Chunk1/j=0 lo, l=0: `is` has advanced to 8 (sc=-32), `m` has rotated to 16
        // (after 4 j-iterations of chunk0, each left-shifting m by 1: 1->2->4->8->16).
        // qs[32] and hmask[0]'s bit4 are both 0 -> bits=0, centered=0-4=-4.
        // v = d*(-32)*(-4) = 128.
        expect_close(out[128], 128.0f, "q3_k chunk1 lo element 128 (m persists across chunks)");
    }

} // namespace

int main()
{
    test_sizes();
    test_q8_0();
    test_q4_0();
    test_q4_k();
    test_q6_k();
    test_q5_k();
    test_q2_k();
    test_q3_k();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All quant block tests passed" << std::endl;
    return 0;
}
