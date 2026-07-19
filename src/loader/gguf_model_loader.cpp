#include "loader/gguf_model_loader.h"
#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/linear_layer.h"
#include "layers/quantized_linear.h"
#include "layers/rms_norm.h"
#include "layers/swiglu.h"
#include "layers/transformer_block.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mini_inference::loader
{

    namespace
    {

        using mini_inference::layers::Embedding;
        using mini_inference::layers::GateActivation;
        using mini_inference::layers::Linear;
        using mini_inference::layers::LinearLayer;
        using mini_inference::layers::MultiHeadAttention;
        using mini_inference::layers::QuantizedLinear;
        using mini_inference::layers::RmsNorm;
        using mini_inference::layers::SwiGLU;
        using mini_inference::layers::TransformerBlock;
        using mini_inference::model::Model;

        // Checks a weight tensor's on-disk shape before copying its data out.
        std::vector<float> required_tensor(const GgufReader &reader, const std::string &name,
                                            const std::vector<std::size_t> &expected_shape)
        {
            const GgufTensorInfo &info = reader.tensor_info(name);
            if (info.shape != expected_shape)
            {
                throw std::invalid_argument("GGUF tensor '" + name + "' has an unexpected shape");
            }
            return reader.tensor_as_f32(name);
        }

        // A missing bias tensor just means "use the zero-bias default".
        std::vector<float> optional_tensor(const GgufReader &reader, const std::string &name,
                                            const std::vector<std::size_t> &expected_shape)
        {
            if (!reader.has_tensor(name))
            {
                return {};
            }
            return required_tensor(reader, name, expected_shape);
        }

        std::string block_prefix(std::size_t layer_index)
        {
            return "blk." + std::to_string(layer_index) + ".";
        }

        // Builds a projection's LinearLayer from its on-disk weight tensor, dispatching
        // to Linear for F32/F16 weights or QuantizedLinear for Q8_0/Q4_0/Q4_K weights -
        // real checkpoints mix these per-tensor, so this decision is made independently
        // for every projection rather than once for the whole checkpoint. `weight_name`'s
        // GGUF shape must be [in_features, out_features] (ne0 = in_features).
        LinearLayer build_linear_layer(const GgufReader &reader, const std::string &weight_name,
                                        const std::optional<std::string> &bias_name,
                                        std::size_t in_features, std::size_t out_features)
        {
            const GgufTensorInfo &info = reader.tensor_info(weight_name);
            if (info.shape != std::vector<std::size_t>{in_features, out_features})
            {
                throw std::invalid_argument("GGUF tensor '" + weight_name + "' has an unexpected shape");
            }

            std::vector<float> bias =
                bias_name.has_value() ? optional_tensor(reader, *bias_name, {out_features}) : std::vector<float>{};

            switch (static_cast<GgmlType>(info.ggml_type))
            {
            case GgmlType::kF32:
            case GgmlType::kF16:
                return LinearLayer(Linear(in_features, out_features, reader.tensor_as_f32(weight_name), std::move(bias)));
            case GgmlType::kQ8_0:
            case GgmlType::kQ4_0:
            case GgmlType::kQ4_K:
            case GgmlType::kQ6_K:
            case GgmlType::kQ5_K:
            case GgmlType::kQ2_K:
            case GgmlType::kQ3_K:
                return LinearLayer(QuantizedLinear(reader.tensor_as_quantized(weight_name), std::move(bias)));
            default:
                throw std::invalid_argument("GGUF tensor '" + weight_name + "' uses unsupported ggml type " +
                                             std::to_string(info.ggml_type));
            }
        }

        // Reads a weight matrix that's always needed as a flat, fully-materialized
        // float32 vector rather than lazily dequantized like QuantizedLinear's per-row
        // scheme - appropriate for the embedding table, which needs a full row-major
        // float array regardless and is only read once at load time, not per forward
        // call. F32/F16 tensors convert directly; quantized tensors are eagerly
        // dequantized in full via QuantizedTensor::dequantize().
        std::vector<float> read_dense_weight_matrix(const GgufReader &reader, const std::string &name)
        {
            const GgufTensorInfo &info = reader.tensor_info(name);
            switch (static_cast<GgmlType>(info.ggml_type))
            {
            case GgmlType::kF32:
            case GgmlType::kF16:
                return reader.tensor_as_f32(name);
            case GgmlType::kQ8_0:
            case GgmlType::kQ4_0:
            case GgmlType::kQ4_K:
            case GgmlType::kQ6_K:
            case GgmlType::kQ5_K:
            case GgmlType::kQ2_K:
            case GgmlType::kQ3_K:
                return reader.tensor_as_quantized(name).dequantize().values();
            default:
                throw std::invalid_argument("GGUF tensor '" + name + "' uses unsupported ggml type " +
                                             std::to_string(info.ggml_type));
            }
        }

        // qwen2 mirrors llama's tensors and metadata under a "qwen2." prefix. gemma also
        // scales the input embedding, offsets RmsNorm by +1, and gates its FFN with GELU.
        bool is_supported_architecture(const std::string &architecture)
        {
            return architecture == "llama" || architecture == "qwen2" || architecture == "gemma";
        }

        // Larger Gemma variants (e.g. gemma-7b) have a head_dim that isn't
        // embedding_length/head_count; reject those instead of silently using a wrong shape.
        void check_gemma_head_dim(const GgufReader &reader, const std::string &metadata_prefix,
                                   std::size_t derived_head_dim)
        {
            const std::string key = metadata_prefix + "attention.key_length";
            if (!reader.has_metadata(key))
            {
                return;
            }
            const std::size_t declared_head_dim = reader.metadata_uint32(key, 0);
            if (declared_head_dim != derived_head_dim)
            {
                throw std::invalid_argument(
                    "gemma head_dim derived from embedding_length/head_count (" + std::to_string(derived_head_dim) +
                    ") does not match " + key + " (" + std::to_string(declared_head_dim) +
                    "); only Gemma variants where head_dim == embedding_length/head_count are supported");
            }
        }

        // Gemma norms as x * (1 + weight); baking the +1 in keeps RmsNorm arch-agnostic.
        std::vector<float> load_norm_weight(const GgufReader &reader, const std::string &name, std::size_t dim,
                                             bool add_one)
        {
            std::vector<float> gamma = required_tensor(reader, name, {dim});
            if (add_one)
            {
                for (float &value : gamma)
                {
                    value += 1.0f;
                }
            }
            return gamma;
        }

    } // namespace

    Model build_model(const GgufReader &reader)
    {
        const std::string architecture = reader.metadata_string("general.architecture");
        if (!is_supported_architecture(architecture))
        {
            throw std::invalid_argument("unsupported GGUF architecture '" + architecture +
                                         "' (supported: llama, qwen2, gemma)");
        }
        const bool is_gemma = architecture == "gemma";
        const std::string metadata_prefix = architecture + ".";

        const std::size_t hidden_dim = reader.metadata_uint32(metadata_prefix + "embedding_length", 0);
        const std::size_t num_layers = reader.metadata_uint32(metadata_prefix + "block_count", 0);
        const std::size_t intermediate_dim = reader.metadata_uint32(metadata_prefix + "feed_forward_length", 0);
        const std::uint32_t num_heads = reader.metadata_uint32(metadata_prefix + "attention.head_count", 0);
        const std::uint32_t num_kv_heads =
            reader.metadata_uint32(metadata_prefix + "attention.head_count_kv", num_heads);
        const float rms_eps = reader.metadata_float(metadata_prefix + "attention.layer_norm_rms_epsilon", 1e-5f);
        const float rope_theta = reader.metadata_float(metadata_prefix + "rope.freq_base", 10000.0f);
        const std::size_t max_position_embeddings = reader.metadata_uint32(metadata_prefix + "context_length", 2048);

        const std::size_t head_dim = num_heads == 0 ? 0 : hidden_dim / num_heads;
        const std::size_t kv_dim = static_cast<std::size_t>(num_kv_heads) * head_dim;

        if (is_gemma)
        {
            check_gemma_head_dim(reader, metadata_prefix, head_dim);
        }

        const GgufTensorInfo &token_embd_info = reader.tensor_info("token_embd.weight");
        if (token_embd_info.shape.size() != 2 || token_embd_info.shape[0] != hidden_dim)
        {
            throw std::invalid_argument("GGUF tensor 'token_embd.weight' has an unexpected shape");
        }
        const std::size_t vocab_size = token_embd_info.shape[1];

        if (reader.has_metadata(metadata_prefix + "vocab_size"))
        {
            const std::size_t declared_vocab_size = reader.metadata_uint32(metadata_prefix + "vocab_size", 0);
            if (declared_vocab_size != vocab_size)
            {
                throw std::invalid_argument(metadata_prefix + "vocab_size metadata (" +
                                             std::to_string(declared_vocab_size) +
                                             ") does not match token_embd.weight's row count (" +
                                             std::to_string(vocab_size) + ")");
            }
        }

        // Unscaled, for a tied lm_head; gemma scales only the input embedding lookup below.
        const std::vector<float> token_embedding = read_dense_weight_matrix(reader, "token_embd.weight");
        std::vector<float> input_embedding = token_embedding;
        if (is_gemma)
        {
            const float normalizer = std::sqrt(static_cast<float>(hidden_dim));
            for (float &value : input_embedding)
            {
                value *= normalizer;
            }
        }
        Embedding embedding(vocab_size, hidden_dim, std::move(input_embedding));

        const GateActivation ffn_activation = is_gemma ? GateActivation::kGelu : GateActivation::kSilu;

        std::vector<TransformerBlock> blocks;
        blocks.reserve(num_layers);
        for (std::size_t layer = 0; layer < num_layers; ++layer)
        {
            const std::string prefix = block_prefix(layer);

            RmsNorm attn_norm(hidden_dim, rms_eps,
                               load_norm_weight(reader, prefix + "attn_norm.weight", hidden_dim, is_gemma));

            MultiHeadAttention attention(
                hidden_dim, num_heads, /*causal=*/true, rope_theta, max_position_embeddings,
                build_linear_layer(reader, prefix + "attn_q.weight", prefix + "attn_q.bias", hidden_dim, hidden_dim),
                build_linear_layer(reader, prefix + "attn_k.weight", prefix + "attn_k.bias", hidden_dim, kv_dim),
                build_linear_layer(reader, prefix + "attn_v.weight", prefix + "attn_v.bias", hidden_dim, kv_dim),
                build_linear_layer(reader, prefix + "attn_output.weight", prefix + "attn_output.bias", hidden_dim,
                                    hidden_dim),
                num_kv_heads);

            RmsNorm ffn_norm(hidden_dim, rms_eps,
                              load_norm_weight(reader, prefix + "ffn_norm.weight", hidden_dim, is_gemma));

            SwiGLU ffn(
                hidden_dim, intermediate_dim,
                build_linear_layer(reader, prefix + "ffn_gate.weight", prefix + "ffn_gate.bias", hidden_dim,
                                    intermediate_dim),
                build_linear_layer(reader, prefix + "ffn_up.weight", prefix + "ffn_up.bias", hidden_dim,
                                    intermediate_dim),
                build_linear_layer(reader, prefix + "ffn_down.weight", prefix + "ffn_down.bias", intermediate_dim,
                                    hidden_dim),
                ffn_activation);

            blocks.emplace_back(std::move(attn_norm), std::move(attention), std::move(ffn_norm), std::move(ffn));
        }

        RmsNorm final_norm(hidden_dim, rms_eps, load_norm_weight(reader, "output_norm.weight", hidden_dim, is_gemma));

        // Tied embeddings: small checkpoints often omit output.weight entirely.
        LinearLayer lm_head = reader.has_tensor("output.weight")
                                   ? build_linear_layer(reader, "output.weight", std::nullopt, hidden_dim, vocab_size)
                                   : LinearLayer(Linear(hidden_dim, vocab_size, token_embedding, {}));

        return Model(std::move(embedding), std::move(blocks), std::move(final_norm), std::move(lm_head));
    }

} // namespace mini_inference::loader
