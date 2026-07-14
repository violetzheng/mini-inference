#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace mini_inference::loader
{

    // GPT-2's reversible byte<->unicode mapping (encoder.py's bytes_to_unicode()):
    // printable bytes map to themselves, the rest map to extra code points from 256
    // up. GGUF's "gpt2" tokenizer metadata stores vocab/merges in this display form.
    class Gpt2ByteEncoding
    {
    public:
        Gpt2ByteEncoding();

        std::string encode(const std::string &raw_bytes) const;

        // Throws std::invalid_argument on malformed UTF-8 or an unrecognized code point.
        std::string decode(const std::string &display_string) const;

    private:
        std::array<std::string, 256> byte_to_display_{};
        std::unordered_map<char32_t, unsigned char> display_codepoint_to_byte_{};
    };

} // namespace mini_inference::loader
