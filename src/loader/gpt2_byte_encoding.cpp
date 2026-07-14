#include "loader/gpt2_byte_encoding.h"

#include <stdexcept>
#include <vector>

namespace mini_inference::loader
{

    namespace
    {

        std::string utf8_encode_codepoint(char32_t codepoint)
        {
            std::string out;
            if (codepoint <= 0x7F)
            {
                out.push_back(static_cast<char>(codepoint));
            }
            else if (codepoint <= 0x7FF)
            {
                out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
                out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
            }
            else if (codepoint <= 0xFFFF)
            {
                out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
                out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
            }
            else
            {
                out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
                out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
            }
            return out;
        }

    } // namespace

    Gpt2ByteEncoding::Gpt2ByteEncoding()
    {
        std::vector<bool> is_self_mapped(256, false);
        for (int b = 33; b <= 126; ++b)
        {
            is_self_mapped[static_cast<std::size_t>(b)] = true;
        }
        for (int b = 161; b <= 172; ++b)
        {
            is_self_mapped[static_cast<std::size_t>(b)] = true;
        }
        for (int b = 174; b <= 255; ++b)
        {
            is_self_mapped[static_cast<std::size_t>(b)] = true;
        }

        char32_t next_extra_codepoint = 256;
        for (int b = 0; b < 256; ++b)
        {
            const char32_t codepoint = is_self_mapped[static_cast<std::size_t>(b)]
                                            ? static_cast<char32_t>(b)
                                            : next_extra_codepoint++;
            byte_to_display_[static_cast<std::size_t>(b)] = utf8_encode_codepoint(codepoint);
            display_codepoint_to_byte_[codepoint] = static_cast<unsigned char>(b);
        }
    }

    std::string Gpt2ByteEncoding::encode(const std::string &raw_bytes) const
    {
        std::string result;
        result.reserve(raw_bytes.size());
        for (unsigned char byte : raw_bytes)
        {
            result += byte_to_display_[byte];
        }
        return result;
    }

    std::string Gpt2ByteEncoding::decode(const std::string &display_string) const
    {
        std::string result;
        std::size_t i = 0;
        const std::size_t n = display_string.size();

        while (i < n)
        {
            const unsigned char lead = static_cast<unsigned char>(display_string[i]);
            std::size_t extra_bytes;
            char32_t codepoint;

            if ((lead & 0x80) == 0x00)
            {
                extra_bytes = 0;
                codepoint = lead;
            }
            else if ((lead & 0xE0) == 0xC0)
            {
                extra_bytes = 1;
                codepoint = lead & 0x1F;
            }
            else if ((lead & 0xF0) == 0xE0)
            {
                extra_bytes = 2;
                codepoint = lead & 0x0F;
            }
            else if ((lead & 0xF8) == 0xF0)
            {
                extra_bytes = 3;
                codepoint = lead & 0x07;
            }
            else
            {
                throw std::invalid_argument("invalid UTF-8 leading byte in GPT-2 token string");
            }

            if (i + 1 + extra_bytes > n)
            {
                throw std::invalid_argument("truncated UTF-8 sequence in GPT-2 token string");
            }

            for (std::size_t k = 1; k <= extra_bytes; ++k)
            {
                const unsigned char continuation = static_cast<unsigned char>(display_string[i + k]);
                if ((continuation & 0xC0) != 0x80)
                {
                    throw std::invalid_argument("invalid UTF-8 continuation byte in GPT-2 token string");
                }
                codepoint = (codepoint << 6) | (continuation & 0x3F);
            }

            const auto it = display_codepoint_to_byte_.find(codepoint);
            if (it == display_codepoint_to_byte_.end())
            {
                throw std::invalid_argument("GPT-2 token string contains a code point with no byte mapping");
            }
            result.push_back(static_cast<char>(it->second));

            i += 1 + extra_bytes;
        }

        return result;
    }

} // namespace mini_inference::loader
