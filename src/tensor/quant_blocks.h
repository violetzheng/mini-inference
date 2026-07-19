#pragma once

#include <cstddef>
#include <cstdint>

namespace mini_inference::tensor
{

    // GGML weight-only quantization formats this engine can decode. Each format packs
    // a fixed number of elements ("block") into a fixed number of bytes, with one or
    // more shared scale factors per block instead of a scale per element.
    enum class QuantFormat
    {
        Q8_0,
        Q4_0,
        Q4_K,
        Q6_K,
        Q5_K,
        Q2_K,
        Q3_K,
    };

    // Number of float elements a single block of `format` decodes into.
    std::size_t block_size(QuantFormat format);

    // Number of bytes a single block of `format` occupies on disk / in memory.
    std::size_t block_byte_size(QuantFormat format);

    // Decodes exactly one block of `format` from `block` (block_byte_size(format) bytes)
    // into `out` (block_size(format) floats). The block layouts mirror GGML's on-disk
    // format exactly:
    //
    //   Q8_0 (32 elements, 34 bytes): fp16 d; int8_t qs[32];
    //     v[i] = qs[i] * d
    //
    //   Q4_0 (32 elements, 18 bytes): fp16 d; uint8_t qs[16];
    //     low nibble of qs[i] is element i, high nibble is element i+16, offset by 8:
    //     v[i]    = ((qs[i] & 0xF) - 8) * d
    //     v[i+16] = ((qs[i] >> 4) - 8) * d,  for i in 0..15
    //
    //   Q4_K (256 elements, 144 bytes): fp16 d; fp16 dmin; uint8_t scales[12]; uint8_t qs[128];
    //     8 sub-blocks of 32 elements, each with its own 6-bit scale/min unpacked from
    //     `scales` (see get_scale_min_k4 in quant_blocks.cpp). Both nibble groups of a
    //     64-element chunk are decoded from the *same* 32 bytes of qs.
    //
    //   Q6_K (256 elements, 210 bytes): uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16];
    //     fp16 d. Unlike every format above, the super-block scale `d` comes *last*, not
    //     first. 2 halves of 128 elements each; within a half, element l (0..31) reads a
    //     6-bit unsigned quant split across ql (low 4 bits) and qh (high 2 bits), one of
    //     4 positions per l (yielding elements l, l+32, l+64, l+96), each position scaled
    //     by its own signed 8-bit sub-scale (2 of the 16 scales per half, one per 16-wide
    //     group): v = d * scale * (q - 32).
    //
    //   Q5_K (256 elements, 176 bytes): fp16 d; fp16 dmin; uint8_t scales[12]; uint8_t qh[32];
    //     uint8_t qs[128]. Same 8-sub-blocks-of-32/get_scale_min_k4 structure as Q4_K, but
    //     each quant gets a 5th bit from qh (one bit per element, indexed 0..255 across the
    //     whole block, reused for both the low- and high-nibble half of each 64-element
    //     qs chunk the way Q4_K's qs is): q = nibble | (qh_bit << 4), range 0..31.
    //     v = d*scale*q - dmin*min (same affine form as Q4_K).
    //
    //   Q2_K (256 elements, 84 bytes): uint8_t scales[16]; uint8_t qs[64]; fp16 d; fp16 dmin.
    //     16 sub-blocks of 16 elements; sub-block j's scale/min come directly from one byte
    //     (scale = scales[j] & 0xF, min = scales[j] >> 4 - no cross-byte unpacking, unlike
    //     Q4_K/Q5_K). Quants are 2 bits, 4 packed per qs byte: sub-block j (0..3 within a
    //     64-element chunk) reads (qs[byte] >> (2*j)) & 3. v = d*scale*q - dmin*min.
    //
    //   Q3_K (256 elements, 110 bytes): uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12];
    //     fp16 d. 16 sub-blocks of 16 elements, each with a *signed* 6-bit scale (-32..31,
    //     `sc - 32`) unpacked from the 12-byte scales array via a 4x uint32_t bit-shuffle
    //     (see quant_blocks.cpp). Quant is 2 bits from qs (same shift-based reader as Q2_K)
    //     combined with a sign adjustment from hmask (a rotating single-bit mask over the
    //     16 sub-blocks): q = (2-bit value) - (hmask bit set ? 0 : 4). v = d * scale * q
    //     (no separate min term, unlike Q2_K/Q4_K/Q5_K).
    void dequantize_block_q8_0(const std::byte *block, float *out);
    void dequantize_block_q4_0(const std::byte *block, float *out);
    void dequantize_block_q4_k(const std::byte *block, float *out);
    void dequantize_block_q6_k(const std::byte *block, float *out);
    void dequantize_block_q5_k(const std::byte *block, float *out);
    void dequantize_block_q2_k(const std::byte *block, float *out);
    void dequantize_block_q3_k(const std::byte *block, float *out);

    // Dispatches to the matching dequantize_block_* for `format`.
    void dequantize_block(QuantFormat format, const std::byte *block, float *out);

} // namespace mini_inference::tensor
