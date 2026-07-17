#pragma once

#include <cstdint>

namespace mini_inference::tensor
{

    // IEEE 754 binary16 -> binary32, handling zero/subnormal/inf/nan. Shared by the
    // GGUF loader (F16 tensors) and quantized block decoding (fp16 block scales).
    float f16_to_f32(std::uint16_t half);

} // namespace mini_inference::tensor
