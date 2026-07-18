#include "tensor/quant_blocks.h"

#include "tensor/half_float.h"

#include <cstring>
#include <stdexcept>

namespace mini_inference::tensor
{

    namespace
    {

        constexpr std::size_t kBlockSizeQ8_0 = 32;
        constexpr std::size_t kBlockByteSizeQ8_0 = 2 + 32; // fp16 d + 32 int8 quants

        constexpr std::size_t kBlockSizeQ4_0 = 32;
        constexpr std::size_t kBlockByteSizeQ4_0 = 2 + 16; // fp16 d + 16 packed nibble bytes

        constexpr std::size_t kBlockSizeQ4_K = 256;
        constexpr std::size_t kBlockByteSizeQ4_K = 2 + 2 + 12 + 128; // fp16 d, fp16 dmin, scales[12], qs[128]

        constexpr std::size_t kBlockSizeQ6_K = 256;
        constexpr std::size_t kBlockByteSizeQ6_K = 128 + 64 + 16 + 2; // ql[128], qh[64], scales[16], fp16 d

        std::uint16_t read_u16(const std::byte *bytes)
        {
            std::uint16_t value;
            std::memcpy(&value, bytes, sizeof(value));
            return value;
        }

        // Unpacks the 6-bit scale and 6-bit min for sub-block `j` (0..7) from Q4_K's
        // 12-byte `scales` array. See quant_blocks.h's Q4_K doc comment for context.
        void get_scale_min_k4(std::size_t j, const std::uint8_t *scales, std::uint8_t &sc, std::uint8_t &m)
        {
            if (j < 4)
            {
                sc = scales[j] & 63;
                m = scales[j + 4] & 63;
            }
            else
            {
                sc = static_cast<std::uint8_t>((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
                m = static_cast<std::uint8_t>((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
            }
        }

    } // namespace

    std::size_t block_size(QuantFormat format)
    {
        switch (format)
        {
        case QuantFormat::Q8_0:
            return kBlockSizeQ8_0;
        case QuantFormat::Q4_0:
            return kBlockSizeQ4_0;
        case QuantFormat::Q4_K:
            return kBlockSizeQ4_K;
        case QuantFormat::Q6_K:
            return kBlockSizeQ6_K;
        }
        throw std::invalid_argument("unknown QuantFormat");
    }

    std::size_t block_byte_size(QuantFormat format)
    {
        switch (format)
        {
        case QuantFormat::Q8_0:
            return kBlockByteSizeQ8_0;
        case QuantFormat::Q4_0:
            return kBlockByteSizeQ4_0;
        case QuantFormat::Q4_K:
            return kBlockByteSizeQ4_K;
        case QuantFormat::Q6_K:
            return kBlockByteSizeQ6_K;
        }
        throw std::invalid_argument("unknown QuantFormat");
    }

    void dequantize_block_q8_0(const std::byte *block, float *out)
    {
        // Single scale for the whole block; each quant is already a signed int8, so
        // dequantizing is just a per-element scale multiply.
        const float d = f16_to_f32(read_u16(block));
        const auto *qs = reinterpret_cast<const std::int8_t *>(block + 2);
        for (std::size_t i = 0; i < kBlockSizeQ8_0; ++i)
        {
            out[i] = static_cast<float>(qs[i]) * d;
        }
    }

    void dequantize_block_q4_0(const std::byte *block, float *out)
    {
        // Two 4-bit quants share each byte (low nibble = element i, high nibble =
        // element i+16); nibbles are unsigned 0..15, so subtract 8 to re-center them.
        const float d = f16_to_f32(read_u16(block));
        const auto *qs = reinterpret_cast<const std::uint8_t *>(block + 2);
        for (std::size_t i = 0; i < kBlockSizeQ4_0 / 2; ++i)
        {
            out[i] = (static_cast<float>(qs[i] & 0x0F) - 8.0f) * d;
            out[i + kBlockSizeQ4_0 / 2] = (static_cast<float>(qs[i] >> 4) - 8.0f) * d;
        }
    }

    void dequantize_block_q4_k(const std::byte *block, float *out)
    {
        // d/dmin scale the per-sub-block 6-bit scale/min unpacked by get_scale_min_k4;
        // dequantized value = d*scale*q - dmin*min (an affine map, not a pure scale,
        // since Q4_K's quants are unsigned 0..15 rather than zero-centered like Q4_0).
        const float d = f16_to_f32(read_u16(block));
        const float dmin = f16_to_f32(read_u16(block + 2));
        const auto *scales = reinterpret_cast<const std::uint8_t *>(block + 4);
        const auto *qs = reinterpret_cast<const std::uint8_t *>(block + 4 + 12);

        // 4 chunks of 64 output elements; each chunk's 32 qs bytes are shared by two
        // sub-blocks (low nibbles -> first 32 elements, high nibbles -> next 32).
        for (std::size_t chunk = 0; chunk < 4; ++chunk)
        {
            const std::size_t is = chunk * 2;
            std::uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is, scales, sc0, m0);
            get_scale_min_k4(is + 1, scales, sc1, m1);

            const float d0 = d * static_cast<float>(sc0);
            const float dm0 = dmin * static_cast<float>(m0);
            const float d1 = d * static_cast<float>(sc1);
            const float dm1 = dmin * static_cast<float>(m1);

            const std::uint8_t *q = qs + chunk * 32;
            float *out_chunk = out + chunk * 64;
            for (std::size_t l = 0; l < 32; ++l)
            {
                out_chunk[l] = d0 * static_cast<float>(q[l] & 0x0F) - dm0;
                out_chunk[32 + l] = d1 * static_cast<float>(q[l] >> 4) - dm1;
            }
        }
    }

    void dequantize_block_q6_k(const std::byte *block, float *out)
    {
        // Field order is reversed vs. every format above: ql/qh/scales come first, and
        // the fp16 super-block scale `d` comes last (byte offset 128+64+16 = 208).
        const auto *ql = reinterpret_cast<const std::uint8_t *>(block);
        const auto *qh = reinterpret_cast<const std::uint8_t *>(block + 128);
        const auto *scales = reinterpret_cast<const std::int8_t *>(block + 128 + 64);
        const float d = f16_to_f32(read_u16(block + 128 + 64 + 16));

        // 2 halves of 128 elements, each with its own 64 ql / 32 qh / 8 scales bytes.
        for (std::size_t half = 0; half < 2; ++half)
        {
            const std::uint8_t *half_ql = ql + half * 64;
            const std::uint8_t *half_qh = qh + half * 32;
            const std::int8_t *half_scales = scales + half * 8;
            float *out_half = out + half * 128;

            for (std::size_t l = 0; l < 32; ++l)
            {
                // Each of l's 4 output positions (l, l+32, l+64, l+96) reassembles a
                // 6-bit unsigned quant from 4 bits of ql + 2 bits of qh, then re-centers
                // it (0..63 -> -32..31). Scales are grouped in 16-wide chunks (is = l/16).
                const std::size_t is = l / 16;
                const auto q1 = static_cast<std::int8_t>((half_ql[l] & 0x0F) | (((half_qh[l] >> 0) & 0x03) << 4)) - 32;
                const auto q2 =
                    static_cast<std::int8_t>((half_ql[l + 32] & 0x0F) | (((half_qh[l] >> 2) & 0x03) << 4)) - 32;
                const auto q3 = static_cast<std::int8_t>((half_ql[l] >> 4) | (((half_qh[l] >> 4) & 0x03) << 4)) - 32;
                const auto q4 =
                    static_cast<std::int8_t>((half_ql[l + 32] >> 4) | (((half_qh[l] >> 6) & 0x03) << 4)) - 32;

                out_half[l] = d * static_cast<float>(half_scales[is + 0]) * static_cast<float>(q1);
                out_half[l + 32] = d * static_cast<float>(half_scales[is + 2]) * static_cast<float>(q2);
                out_half[l + 64] = d * static_cast<float>(half_scales[is + 4]) * static_cast<float>(q3);
                out_half[l + 96] = d * static_cast<float>(half_scales[is + 6]) * static_cast<float>(q4);
            }
        }
    }

    void dequantize_block(QuantFormat format, const std::byte *block, float *out)
    {
        switch (format)
        {
        case QuantFormat::Q8_0:
            dequantize_block_q8_0(block, out);
            return;
        case QuantFormat::Q4_0:
            dequantize_block_q4_0(block, out);
            return;
        case QuantFormat::Q4_K:
            dequantize_block_q4_k(block, out);
            return;
        case QuantFormat::Q6_K:
            dequantize_block_q6_k(block, out);
            return;
        }
        throw std::invalid_argument("unknown QuantFormat");
    }

} // namespace mini_inference::tensor
