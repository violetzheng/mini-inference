#include "loader/gguf_reader.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace mini_inference::loader
{

    namespace
    {

        std::size_t align_up(std::size_t value, std::size_t alignment)
        {
            if (alignment == 0)
            {
                return value;
            }
            return ((value + alignment - 1) / alignment) * alignment;
        }

        // IEEE 754 binary16 -> binary32, handling zero/subnormal/inf/nan.
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

        std::size_t element_count_of(const std::vector<std::size_t> &shape)
        {
            std::size_t count = 1;
            for (std::size_t dim : shape)
            {
                count *= dim;
            }
            return count;
        }

    } // namespace

    GgufReader::GgufReader(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
        {
            throw std::invalid_argument("could not open GGUF file: " + path);
        }

        const std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        buffer_.resize(static_cast<std::size_t>(size));
        if (size > 0 && !file.read(reinterpret_cast<char *>(buffer_.data()), size))
        {
            throw std::invalid_argument("failed to read GGUF file: " + path);
        }

        parse();
    }

    GgufReader::GgufReader(std::vector<std::uint8_t> buffer) : buffer_(std::move(buffer))
    {
        parse();
    }

    void GgufReader::require(std::size_t offset, std::size_t num_bytes) const
    {
        if (offset > buffer_.size() || num_bytes > buffer_.size() - offset)
        {
            throw std::invalid_argument("GGUF file is truncated or corrupt");
        }
    }

    template <typename T>
    T GgufReader::read_pod(std::size_t &offset) const
    {
        require(offset, sizeof(T));
        T value;
        std::memcpy(&value, buffer_.data() + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }

    std::string GgufReader::read_string(std::size_t &offset) const
    {
        const std::uint64_t length = read_pod<std::uint64_t>(offset);
        require(offset, static_cast<std::size_t>(length));
        std::string value(reinterpret_cast<const char *>(buffer_.data() + offset), static_cast<std::size_t>(length));
        offset += static_cast<std::size_t>(length);
        return value;
    }

    GgufScalar GgufReader::read_scalar(std::size_t &offset, GgufValueType type) const
    {
        switch (type)
        {
        case GgufValueType::kUInt8:
            return read_pod<std::uint8_t>(offset);
        case GgufValueType::kInt8:
            return read_pod<std::int8_t>(offset);
        case GgufValueType::kUInt16:
            return read_pod<std::uint16_t>(offset);
        case GgufValueType::kInt16:
            return read_pod<std::int16_t>(offset);
        case GgufValueType::kUInt32:
            return read_pod<std::uint32_t>(offset);
        case GgufValueType::kInt32:
            return read_pod<std::int32_t>(offset);
        case GgufValueType::kFloat32:
            return read_pod<float>(offset);
        case GgufValueType::kBool:
            return read_pod<std::uint8_t>(offset) != 0;
        case GgufValueType::kString:
            return read_string(offset);
        case GgufValueType::kUInt64:
            return read_pod<std::uint64_t>(offset);
        case GgufValueType::kInt64:
            return read_pod<std::int64_t>(offset);
        case GgufValueType::kFloat64:
            return read_pod<double>(offset);
        case GgufValueType::kArray:
            throw std::invalid_argument("nested GGUF arrays are not supported");
        default:
            throw std::invalid_argument("unknown GGUF metadata value type");
        }
    }

    GgufValue GgufReader::read_metadata_value(std::size_t &offset) const
    {
        const auto type = static_cast<GgufValueType>(read_pod<std::uint32_t>(offset));

        if (type != GgufValueType::kArray)
        {
            GgufValue value;
            value.type = type;
            value.is_array = false;
            value.scalar = read_scalar(offset, type);
            return value;
        }

        const auto element_type = static_cast<GgufValueType>(read_pod<std::uint32_t>(offset));
        if (element_type == GgufValueType::kArray)
        {
            throw std::invalid_argument("nested GGUF arrays are not supported");
        }
        const std::uint64_t length = read_pod<std::uint64_t>(offset);

        GgufValue value;
        value.type = element_type;
        value.is_array = true;
        value.array.reserve(static_cast<std::size_t>(length));
        for (std::uint64_t i = 0; i < length; ++i)
        {
            value.array.push_back(read_scalar(offset, element_type));
        }
        return value;
    }

    void GgufReader::parse()
    {
        std::size_t offset = 0;

        require(offset, 4);
        if (buffer_[0] != 'G' || buffer_[1] != 'G' || buffer_[2] != 'U' || buffer_[3] != 'F')
        {
            throw std::invalid_argument("not a GGUF file (bad magic)");
        }
        offset += 4;

        version_ = read_pod<std::uint32_t>(offset);
        if (version_ != 2 && version_ != 3)
        {
            throw std::invalid_argument("unsupported GGUF version " + std::to_string(version_) +
                                         " (only versions 2 and 3 are supported)");
        }

        const std::uint64_t tensor_count = read_pod<std::uint64_t>(offset);
        const std::uint64_t metadata_kv_count = read_pod<std::uint64_t>(offset);

        for (std::uint64_t i = 0; i < metadata_kv_count; ++i)
        {
            std::string key = read_string(offset);
            metadata_[std::move(key)] = read_metadata_value(offset);
        }

        struct PendingTensor
        {
            GgufTensorInfo info;
            std::uint64_t relative_offset;
        };
        std::vector<PendingTensor> pending;
        pending.reserve(static_cast<std::size_t>(tensor_count));

        for (std::uint64_t i = 0; i < tensor_count; ++i)
        {
            GgufTensorInfo info;
            info.name = read_string(offset);

            const std::uint32_t n_dims = read_pod<std::uint32_t>(offset);
            info.shape.resize(n_dims);
            for (std::uint32_t d = 0; d < n_dims; ++d)
            {
                info.shape[d] = static_cast<std::size_t>(read_pod<std::uint64_t>(offset));
            }

            info.ggml_type = read_pod<std::uint32_t>(offset);
            const std::uint64_t relative_offset = read_pod<std::uint64_t>(offset);
            info.element_count = element_count_of(info.shape);

            pending.push_back(PendingTensor{std::move(info), relative_offset});
        }

        std::uint32_t alignment = 32;
        auto alignment_it = metadata_.find("general.alignment");
        if (alignment_it != metadata_.end() && !alignment_it->second.is_array)
        {
            alignment = static_cast<std::uint32_t>(as_integer(alignment_it->second, "general.alignment"));
        }

        const std::size_t data_section_start = align_up(offset, alignment);

        for (auto &pending_tensor : pending)
        {
            pending_tensor.info.data_offset = data_section_start + static_cast<std::size_t>(pending_tensor.relative_offset);
            tensors_[pending_tensor.info.name] = std::move(pending_tensor.info);
        }
    }

    std::uint32_t GgufReader::version() const
    {
        return version_;
    }

    bool GgufReader::has_metadata(const std::string &key) const
    {
        return metadata_.find(key) != metadata_.end();
    }

    const GgufValue &GgufReader::metadata(const std::string &key) const
    {
        const auto it = metadata_.find(key);
        if (it == metadata_.end())
        {
            throw std::out_of_range("GGUF file has no metadata key '" + key + "'");
        }
        return it->second;
    }

    double GgufReader::as_float(const GgufValue &value, const std::string &key) const
    {
        if (value.is_array)
        {
            throw std::invalid_argument("GGUF metadata key '" + key + "' is an array, not a scalar float");
        }
        return std::visit(
            [&key](auto &&arg) -> double
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
                {
                    return static_cast<double>(arg);
                }
                else
                {
                    throw std::invalid_argument("GGUF metadata key '" + key + "' is not a float");
                }
            },
            value.scalar);
    }

    std::uint64_t GgufReader::as_integer(const GgufValue &value, const std::string &key) const
    {
        if (value.is_array)
        {
            throw std::invalid_argument("GGUF metadata key '" + key + "' is an array, not a scalar integer");
        }
        return std::visit(
            [&key](auto &&arg) -> std::uint64_t
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>)
                {
                    return static_cast<std::uint64_t>(arg);
                }
                else
                {
                    throw std::invalid_argument("GGUF metadata key '" + key + "' is not an integer");
                }
            },
            value.scalar);
    }

    std::string GgufReader::metadata_string(const std::string &key) const
    {
        const GgufValue &value = metadata(key);
        if (value.is_array || !std::holds_alternative<std::string>(value.scalar))
        {
            throw std::invalid_argument("GGUF metadata key '" + key + "' is not a string");
        }
        return std::get<std::string>(value.scalar);
    }

    std::uint32_t GgufReader::metadata_uint32(const std::string &key, std::uint32_t default_value) const
    {
        if (!has_metadata(key))
        {
            return default_value;
        }
        return static_cast<std::uint32_t>(as_integer(metadata(key), key));
    }

    float GgufReader::metadata_float(const std::string &key, float default_value) const
    {
        if (!has_metadata(key))
        {
            return default_value;
        }
        return static_cast<float>(as_float(metadata(key), key));
    }

    std::vector<std::string> GgufReader::metadata_string_array(const std::string &key) const
    {
        const GgufValue &value = metadata(key);
        if (!value.is_array)
        {
            throw std::invalid_argument("GGUF metadata key '" + key + "' is not an array");
        }

        std::vector<std::string> result;
        result.reserve(value.array.size());
        for (const GgufScalar &element : value.array)
        {
            if (!std::holds_alternative<std::string>(element))
            {
                throw std::invalid_argument("GGUF metadata key '" + key + "' is not a string array");
            }
            result.push_back(std::get<std::string>(element));
        }
        return result;
    }

    bool GgufReader::has_tensor(const std::string &name) const
    {
        return tensors_.find(name) != tensors_.end();
    }

    const GgufTensorInfo &GgufReader::tensor_info(const std::string &name) const
    {
        const auto it = tensors_.find(name);
        if (it == tensors_.end())
        {
            throw std::out_of_range("GGUF file has no tensor named '" + name + "'");
        }
        return it->second;
    }

    std::vector<float> GgufReader::tensor_as_f32(const std::string &name) const
    {
        const GgufTensorInfo &info = tensor_info(name);

        constexpr std::uint32_t kGgmlTypeF32 = 0;
        constexpr std::uint32_t kGgmlTypeF16 = 1;

        if (info.ggml_type == kGgmlTypeF32)
        {
            require(info.data_offset, info.element_count * sizeof(float));
            std::vector<float> values(info.element_count);
            std::memcpy(values.data(), buffer_.data() + info.data_offset, info.element_count * sizeof(float));
            return values;
        }

        if (info.ggml_type == kGgmlTypeF16)
        {
            require(info.data_offset, info.element_count * sizeof(std::uint16_t));
            std::vector<float> values(info.element_count);
            for (std::size_t i = 0; i < info.element_count; ++i)
            {
                std::uint16_t half;
                std::memcpy(&half, buffer_.data() + info.data_offset + i * sizeof(std::uint16_t), sizeof(std::uint16_t));
                values[i] = f16_to_f32(half);
            }
            return values;
        }

        throw std::invalid_argument("GGUF tensor '" + name + "' uses unsupported ggml type " +
                                     std::to_string(info.ggml_type) + " (only F32=0 and F16=1 are supported)");
    }

} // namespace mini_inference::loader
