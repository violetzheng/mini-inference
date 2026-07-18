#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "tensor/quantized_tensor.h"

namespace mini_inference::loader
{

    // GGML on-disk tensor element type tags (matches ggml's `enum ggml_type` values).
    // Only F32/F16/Q8_0/Q4_0/Q4_K are currently decodable (see tensor_as_f32 /
    // tensor_as_quantized); the rest are named here so unsupported-type errors can
    // report a recognizable name instead of a bare integer.
    enum class GgmlType : std::uint32_t
    {
        kF32 = 0,
        kF16 = 1,
        kQ4_0 = 2,
        kQ4_1 = 3,
        kQ5_0 = 6,
        kQ5_1 = 7,
        kQ8_0 = 8,
        kQ8_1 = 9,
        kQ2_K = 10,
        kQ3_K = 11,
        kQ4_K = 12,
        kQ5_K = 13,
        kQ6_K = 14,
        kQ8_K = 15,
    };

    // GGUF metadata value type tags (matches the on-disk uint32 enum).
    enum class GgufValueType : std::uint32_t
    {
        kUInt8 = 0,
        kInt8 = 1,
        kUInt16 = 2,
        kInt16 = 3,
        kUInt32 = 4,
        kInt32 = 5,
        kFloat32 = 6,
        kBool = 7,
        kString = 8,
        kArray = 9,
        kUInt64 = 10,
        kInt64 = 11,
        kFloat64 = 12,
    };

    // A single non-array GGUF metadata value.
    using GgufScalar = std::variant<std::uint8_t, std::int8_t, std::uint16_t, std::int16_t,
                                     std::uint32_t, std::int32_t, float, bool, std::string,
                                     std::uint64_t, std::int64_t, double>;

    // Either a scalar or a homogeneous array of scalars; nested arrays aren't supported.
    struct GgufValue
    {
        GgufValueType type{GgufValueType::kUInt8}; // element type when is_array, else the scalar's type
        bool is_array{false};
        GgufScalar scalar{};
        std::vector<GgufScalar> array{};
    };

    struct GgufTensorInfo
    {
        std::string name;
        std::vector<std::size_t> shape; // ne[], ne[0] is the fastest-varying/contiguous dimension
        std::uint32_t ggml_type{0};
        std::size_t data_offset{0}; // absolute byte offset into the reader's buffer
        std::size_t element_count{0};
    };

    // Parses the GGUF container format (header, metadata, tensor-info table, tensor
    // data); knows nothing about "llama" or "gpt2" specifically, see
    // gguf_model_loader.h / gguf_tokenizer_loader.h for that. Little-endian, GGUF
    // v2/v3 only, whole file read into memory up front.
    class GgufReader
    {
    public:
        explicit GgufReader(const std::string &path);
        explicit GgufReader(std::vector<std::uint8_t> buffer);

        std::uint32_t version() const;

        bool has_metadata(const std::string &key) const;
        const GgufValue &metadata(const std::string &key) const;

        std::string metadata_string(const std::string &key) const;
        std::uint32_t metadata_uint32(const std::string &key, std::uint32_t default_value) const;
        float metadata_float(const std::string &key, float default_value) const;
        std::vector<std::string> metadata_string_array(const std::string &key) const;
        std::vector<float> metadata_float_array(const std::string &key) const;

        bool has_tensor(const std::string &name) const;
        const GgufTensorInfo &tensor_info(const std::string &name) const;

        // Converts to float32 from F32/F16; throws for any other ggml_type.
        std::vector<float> tensor_as_f32(const std::string &name) const;

        // Copies the tensor's on-disk quantized blocks as-is (no dequantization) into a
        // QuantizedTensor; throws for any ggml_type other than Q8_0/Q4_0/Q4_K.
        mini_inference::tensor::QuantizedTensor tensor_as_quantized(const std::string &name) const;

    private:
        void parse();
        void require(std::size_t offset, std::size_t num_bytes) const;

        template <typename T>
        T read_pod(std::size_t &offset) const;
        std::string read_string(std::size_t &offset) const;
        GgufScalar read_scalar(std::size_t &offset, GgufValueType type) const;
        GgufValue read_metadata_value(std::size_t &offset) const;

        double as_float(const GgufValue &value, const std::string &key) const;
        std::uint64_t as_integer(const GgufValue &value, const std::string &key) const;

        std::vector<std::uint8_t> buffer_;
        std::uint32_t version_{0};
        std::unordered_map<std::string, GgufValue> metadata_{};
        std::unordered_map<std::string, GgufTensorInfo> tensors_{};
    };

} // namespace mini_inference::loader
