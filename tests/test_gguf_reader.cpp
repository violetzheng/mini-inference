#include "loader/gguf_reader.h"
#include "gguf_test_helpers.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using mini_inference::loader::GgufReader;
using mini_inference::tests::GgufBufferBuilder;

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

    template <typename Fn>
    bool throws_invalid_argument(Fn &&fn)
    {
        try
        {
            fn();
        }
        catch (const std::invalid_argument &)
        {
            return true;
        }
        catch (...)
        {
            return false;
        }
        return false;
    }

    template <typename Fn>
    bool throws_out_of_range(Fn &&fn)
    {
        try
        {
            fn();
        }
        catch (const std::out_of_range &)
        {
            return true;
        }
        catch (...)
        {
            return false;
        }
        return false;
    }

} // namespace

int main()
{
    // --- metadata round trip ---
    {
        GgufBufferBuilder builder;
        builder.add_uint32_kv("u32_key", 42);
        builder.add_float32_kv("f32_key", 3.5f);
        builder.add_string_kv("str_key", "hello gguf");
        builder.add_string_array_kv("arr_key", {"a", "bb", "ccc"});

        GgufReader reader(builder.build());

        expect(reader.version() == 3, "version is 3");
        expect(reader.has_metadata("u32_key"), "has_metadata true for present key");
        expect(!reader.has_metadata("missing_key"), "has_metadata false for absent key");
        expect(reader.metadata_uint32("u32_key", 0) == 42, "uint32 metadata round-trips");
        expect(reader.metadata_uint32("missing_key", 7) == 7, "uint32 metadata missing key returns default");
        expect(reader.metadata_float("f32_key", 0.0f) == 3.5f, "float32 metadata round-trips");
        expect(reader.metadata_string("str_key") == "hello gguf", "string metadata round-trips");

        const auto arr = reader.metadata_string_array("arr_key");
        expect(arr.size() == 3 && arr[0] == "a" && arr[1] == "bb" && arr[2] == "ccc",
               "string array metadata round-trips");

        expect(throws_out_of_range([&]
                                    { (void)reader.metadata("missing_key"); }),
               "metadata() throws out_of_range for missing key");
        expect(throws_invalid_argument([&]
                                        { (void)reader.metadata_string("u32_key"); }),
               "metadata_string throws on type mismatch");
    }

    // --- tensor info + F32 round trip ---
    {
        GgufBufferBuilder builder;
        const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        builder.add_tensor_f32("weight", {3, 2}, values); // ne = [3, 2]: in=3, out=2

        GgufReader reader(builder.build());

        expect(reader.has_tensor("weight"), "has_tensor true for present tensor");
        expect(!reader.has_tensor("missing"), "has_tensor false for absent tensor");

        const auto &info = reader.tensor_info("weight");
        expect(info.shape.size() == 2 && info.shape[0] == 3 && info.shape[1] == 2, "tensor shape round-trips");
        expect(info.ggml_type == 0, "tensor type is F32");
        expect(info.element_count == 6, "tensor element_count is product of shape");

        const auto read_back = reader.tensor_as_f32("weight");
        expect(read_back == values, "tensor_as_f32 exact match for F32 data");

        expect(throws_out_of_range([&]
                                    { (void)reader.tensor_info("missing"); }),
               "tensor_info throws out_of_range for missing tensor");
    }

    // --- F16 dequantization ---
    {
        GgufBufferBuilder builder;
        // 0x3C00 = 1.0f, 0xC000 = -2.0f, 0x0000 = 0.0f
        builder.add_tensor_f16_raw("half", {3}, {0x3C00, 0xC000, 0x0000});

        GgufReader reader(builder.build());
        const auto values = reader.tensor_as_f32("half");
        expect(values.size() == 3, "f16 tensor has 3 elements");
        expect(values[0] == 1.0f, "f16 0x3C00 decodes to 1.0f");
        expect(values[1] == -2.0f, "f16 0xC000 decodes to -2.0f");
        expect(values[2] == 0.0f, "f16 0x0000 decodes to 0.0f");
    }

    // --- unsupported quantization type ---
    {
        GgufBufferBuilder builder;
        builder.add_tensor_unsupported_type("quant", {4}, /*ggml_type=*/2, /*byte_count=*/0);
        GgufReader reader(builder.build());
        expect(throws_invalid_argument([&]
                                        { (void)reader.tensor_as_f32("quant"); }),
               "tensor_as_f32 throws on unsupported ggml_type");
    }

    // --- alignment override round trip ---
    {
        GgufBufferBuilder builder;
        builder.add_uint32_kv("general.alignment", 64);
        builder.add_string_kv("pad_key", "x"); // nudge the header off a 64-byte boundary
        const std::vector<float> values = {9.0f, -9.0f};
        builder.add_tensor_f32("aligned_weight", {2}, values);

        GgufReader reader(builder.build(/*alignment=*/64));
        expect(reader.tensor_as_f32("aligned_weight") == values,
               "tensor data round-trips correctly with a non-default alignment");
    }

    // --- malformed files ---
    {
        GgufBufferBuilder builder;
        builder.add_tensor_f32("w", {2}, {1.0f, 2.0f});
        auto buffer = builder.build();

        {
            auto bad = buffer;
            bad[0] = 'X';
            expect(throws_invalid_argument([&]
                                            { GgufReader r(bad); }),
                   "bad magic throws invalid_argument");
        }

        {
            auto bad = buffer;
            bad[4] = 1;
            bad[5] = 0;
            bad[6] = 0;
            bad[7] = 0;
            expect(throws_invalid_argument([&]
                                            { GgufReader r(bad); }),
                   "GGUF version 1 throws invalid_argument");
        }

        {
            std::vector<std::uint8_t> bad(buffer.begin(), buffer.begin() + 6);
            expect(throws_invalid_argument([&]
                                            { GgufReader r(bad); }),
                   "truncated header throws invalid_argument");
        }

        {
            // "w"'s 8 bytes sit inside a 32-byte padded section; strip more than
            // the padding to actually cut into the real data.
            auto bad = buffer;
            bad.resize(bad.size() - 28);
            GgufReader r(bad);
            expect(throws_invalid_argument([&]
                                            { (void)r.tensor_as_f32("w"); }),
                   "reading tensor data past buffer end throws invalid_argument");
        }
    }

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All gguf_reader tests passed" << std::endl;
    return 0;
}
