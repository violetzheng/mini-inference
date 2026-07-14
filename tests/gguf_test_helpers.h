#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace mini_inference::tests
{

    // Hand-assembles synthetic GGUF byte buffers for tests. Always writes version 3.
    class GgufBufferBuilder
    {
    public:
        void add_uint32_kv(const std::string &key, std::uint32_t value)
        {
            append_string(metadata_, key);
            append_u32(metadata_, 4); // GGUF_METADATA_VALUE_TYPE_UINT32
            append_u32(metadata_, value);
            ++metadata_count_;
        }

        void add_float32_kv(const std::string &key, float value)
        {
            append_string(metadata_, key);
            append_u32(metadata_, 6); // FLOAT32
            append_f32(metadata_, value);
            ++metadata_count_;
        }

        void add_string_kv(const std::string &key, const std::string &value)
        {
            append_string(metadata_, key);
            append_u32(metadata_, 8); // STRING
            append_string(metadata_, value);
            ++metadata_count_;
        }

        void add_string_array_kv(const std::string &key, const std::vector<std::string> &values)
        {
            append_string(metadata_, key);
            append_u32(metadata_, 9); // ARRAY
            append_u32(metadata_, 8); // element type STRING
            append_u64(metadata_, values.size());
            for (const std::string &value : values)
            {
                append_string(metadata_, value);
            }
            ++metadata_count_;
        }

        void add_tensor_f32(const std::string &name, const std::vector<std::size_t> &shape,
                             const std::vector<float> &values)
        {
            add_tensor_info(name, shape, /*ggml_type=*/0);
            for (float value : values)
            {
                append_f32(tensor_data_, value);
            }
            pad_tensor_data();
        }

        void add_tensor_f16_raw(const std::string &name, const std::vector<std::size_t> &shape,
                                 const std::vector<std::uint16_t> &half_bits)
        {
            add_tensor_info(name, shape, /*ggml_type=*/1);
            for (std::uint16_t bits : half_bits)
            {
                append_u16(tensor_data_, bits);
            }
            pad_tensor_data();
        }

        // Writes a tensor claiming an unsupported ggml_type, for negative tests.
        void add_tensor_unsupported_type(const std::string &name, const std::vector<std::size_t> &shape,
                                          std::uint32_t ggml_type, std::size_t byte_count)
        {
            add_tensor_info(name, shape, ggml_type);
            tensor_data_.insert(tensor_data_.end(), byte_count, std::uint8_t{0});
            pad_tensor_data();
        }

        std::vector<std::uint8_t> build(std::uint32_t alignment = 32) const
        {
            std::vector<std::uint8_t> buffer;
            append_bytes(buffer, "GGUF", 4);
            append_u32(buffer, 3); // version
            append_u64(buffer, tensor_count_);
            append_u64(buffer, metadata_count_);
            buffer.insert(buffer.end(), metadata_.begin(), metadata_.end());
            buffer.insert(buffer.end(), tensor_infos_.begin(), tensor_infos_.end());

            const std::size_t unaligned = buffer.size();
            const std::size_t aligned = ((unaligned + alignment - 1) / alignment) * alignment;
            buffer.resize(aligned, 0);

            buffer.insert(buffer.end(), tensor_data_.begin(), tensor_data_.end());
            return buffer;
        }

    private:
        void add_tensor_info(const std::string &name, const std::vector<std::size_t> &shape, std::uint32_t ggml_type)
        {
            append_string(tensor_infos_, name);
            append_u32(tensor_infos_, static_cast<std::uint32_t>(shape.size()));
            for (std::size_t dim : shape)
            {
                append_u64(tensor_infos_, static_cast<std::uint64_t>(dim));
            }
            append_u32(tensor_infos_, ggml_type);
            append_u64(tensor_infos_, tensor_data_.size()); // offset relative to data section start
            ++tensor_count_;
        }

        void pad_tensor_data()
        {
            // Keeps each tensor's start 32-byte aligned, like a real GGUF file.
            while (tensor_data_.size() % 32 != 0)
            {
                tensor_data_.push_back(0);
            }
        }

        static void append_bytes(std::vector<std::uint8_t> &out, const char *data, std::size_t count)
        {
            out.insert(out.end(), reinterpret_cast<const std::uint8_t *>(data),
                       reinterpret_cast<const std::uint8_t *>(data) + count);
        }

        template <typename T>
        static void append_pod(std::vector<std::uint8_t> &out, T value)
        {
            std::uint8_t bytes[sizeof(T)];
            std::memcpy(bytes, &value, sizeof(T));
            out.insert(out.end(), bytes, bytes + sizeof(T));
        }

        static void append_u16(std::vector<std::uint8_t> &out, std::uint16_t value) { append_pod(out, value); }
        static void append_u32(std::vector<std::uint8_t> &out, std::uint32_t value) { append_pod(out, value); }
        static void append_u64(std::vector<std::uint8_t> &out, std::uint64_t value) { append_pod(out, value); }
        static void append_f32(std::vector<std::uint8_t> &out, float value) { append_pod(out, value); }

        static void append_string(std::vector<std::uint8_t> &out, const std::string &value)
        {
            append_u64(out, static_cast<std::uint64_t>(value.size()));
            append_bytes(out, value.data(), value.size());
        }

        std::vector<std::uint8_t> metadata_{};
        std::uint64_t metadata_count_{0};
        std::vector<std::uint8_t> tensor_infos_{};
        std::uint64_t tensor_count_{0};
        std::vector<std::uint8_t> tensor_data_{};
    };

} // namespace mini_inference::tests
