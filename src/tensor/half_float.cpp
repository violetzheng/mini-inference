#include "tensor/half_float.h"

#include <cstring>

namespace mini_inference::tensor
{

    float f16_to_f32(std::uint16_t half)
    {
        const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000) << 16;
        std::uint32_t exponent = (half >> 10) & 0x1F;
        std::uint32_t mantissa = half & 0x3FF;
        std::uint32_t bits;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                bits = sign;
            }
            else
            {
                exponent = 127 - 15 + 1;
                while ((mantissa & 0x400) == 0)
                {
                    mantissa <<= 1;
                    --exponent;
                }
                mantissa &= 0x3FF;
                bits = sign | (exponent << 23) | (mantissa << 13);
            }
        }
        else if (exponent == 0x1F)
        {
            bits = sign | 0x7F800000u | (mantissa << 13);
        }
        else
        {
            bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
        }

        float result;
        std::memcpy(&result, &bits, sizeof(float));
        return result;
    }

} // namespace mini_inference::tensor
