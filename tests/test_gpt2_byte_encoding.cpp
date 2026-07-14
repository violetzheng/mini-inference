#include "loader/gpt2_byte_encoding.h"

#include <iostream>
#include <stdexcept>
#include <string>

using mini_inference::loader::Gpt2ByteEncoding;

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

} // namespace

int main()
{
    const Gpt2ByteEncoding encoding;

    bool all_bytes_round_trip = true;
    for (int byte = 0; byte < 256; ++byte)
    {
        const std::string original(1, static_cast<char>(byte));
        const std::string round_tripped = encoding.decode(encoding.encode(original));
        if (round_tripped != original)
        {
            all_bytes_round_trip = false;
            break;
        }
    }
    expect(all_bytes_round_trip, "decode(encode(byte)) round-trips for every byte value 0-255");

    // Byte 0x20 (space) maps to code point U+0120 ('Ġ'), UTF-8 bytes 0xC4 0xA0.
    const std::string space_encoded = encoding.encode(" ");
    expect(space_encoded.size() == 2 && static_cast<unsigned char>(space_encoded[0]) == 0xC4 &&
               static_cast<unsigned char>(space_encoded[1]) == 0xA0,
           "space byte encodes to U+0120 ('Ġ')");

    // Printable ASCII bytes are self-mapped (encode to themselves, single byte).
    expect(encoding.encode("A") == "A", "printable ASCII byte encodes to itself");
    expect(encoding.encode("hello") == "hello", "printable ASCII string encodes to itself");

    bool threw_bad_leading_byte = false;
    try
    {
        (void)encoding.decode("\xFF");
    }
    catch (const std::invalid_argument &)
    {
        threw_bad_leading_byte = true;
    }
    expect(threw_bad_leading_byte, "decode throws on an invalid UTF-8 leading byte");

    // Code point 1000 (0x3E8) is valid UTF-8 (bytes 0xCF 0xA8) but has no byte mapping.
    bool threw_unmapped_codepoint = false;
    try
    {
        const std::string unmapped_codepoint = "\xCF\xA8";
        (void)encoding.decode(unmapped_codepoint);
    }
    catch (const std::invalid_argument &)
    {
        threw_unmapped_codepoint = true;
    }
    expect(threw_unmapped_codepoint, "decode throws on a valid but unmapped code point");

    bool threw_truncated_sequence = false;
    try
    {
        const std::string truncated = "\xC4"; // leading byte of a 2-byte sequence, missing continuation
        (void)encoding.decode(truncated);
    }
    catch (const std::invalid_argument &)
    {
        threw_truncated_sequence = true;
    }
    expect(threw_truncated_sequence, "decode throws on a truncated UTF-8 sequence");

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All gpt2_byte_encoding tests passed" << std::endl;
    return 0;
}
