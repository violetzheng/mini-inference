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
    void dequantize_block_q8_0(const std::byte *block, float *out);
    void dequantize_block_q4_0(const std::byte *block, float *out);
    void dequantize_block_q4_k(const std::byte *block, float *out);

    // Dispatches to the matching dequantize_block_* for `format`.
    void dequantize_block(QuantFormat format, const std::byte *block, float *out);

} // namespace mini_inference::tensor
