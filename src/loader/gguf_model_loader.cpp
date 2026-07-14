#include "loader/gguf_model_loader.h"
#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/rms_norm.h"
#include "layers/swiglu.h"
#include "layers/transformer_block.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mini_inference::loader
{

    namespace
    {

        using mini_inference::layers::Embedding;
        using mini_inference::layers::Linear;
        using mini_inference::layers::MultiHeadAttention;
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

    } // namespace

    Model build_model(const GgufReader &reader)
    {
        const std::string architecture = reader.metadata_string("general.architecture");
        if (architecture != "llama")
        {
            throw std::invalid_argument("unsupported GGUF architecture '" + architecture +
                                         "' (only 'llama' is supported)");
        }

        const std::size_t hidden_dim = reader.metadata_uint32("llama.embedding_length", 0);
        const std::size_t num_layers = reader.metadata_uint32("llama.block_count", 0);
        const std::size_t intermediate_dim = reader.metadata_uint32("llama.feed_forward_length", 0);
        const std::uint32_t num_heads = reader.metadata_uint32("llama.attention.head_count", 0);
        const std::uint32_t num_kv_heads = reader.metadata_uint32("llama.attention.head_count_kv", num_heads);
        const float rms_eps = reader.metadata_float("llama.attention.layer_norm_rms_epsilon", 1e-5f);
        const float rope_theta = reader.metadata_float("llama.rope.freq_base", 10000.0f);
        const std::size_t max_position_embeddings = reader.metadata_uint32("llama.context_length", 2048);

        if (num_kv_heads != num_heads)
        {
            throw std::invalid_argument("GGUF checkpoint uses grouped-query attention (head_count=" +
                                         std::to_string(num_heads) + ", head_count_kv=" +
                                         std::to_string(num_kv_heads) +
                                         "), which MultiHeadAttention does not support");
        }

        const GgufTensorInfo &token_embd_info = reader.tensor_info("token_embd.weight");
        if (token_embd_info.shape.size() != 2 || token_embd_info.shape[0] != hidden_dim)
        {
            throw std::invalid_argument("GGUF tensor 'token_embd.weight' has an unexpected shape");
        }
        const std::size_t vocab_size = token_embd_info.shape[1];

        if (reader.has_metadata("llama.vocab_size"))
        {
            const std::size_t declared_vocab_size = reader.metadata_uint32("llama.vocab_size", 0);
            if (declared_vocab_size != vocab_size)
            {
                throw std::invalid_argument("llama.vocab_size metadata (" + std::to_string(declared_vocab_size) +
                                             ") does not match token_embd.weight's row count (" +
                                             std::to_string(vocab_size) + ")");
            }
        }

        const std::vector<float> token_embedding = reader.tensor_as_f32("token_embd.weight");
        Embedding embedding(vocab_size, hidden_dim, token_embedding);

        std::vector<TransformerBlock> blocks;
        blocks.reserve(num_layers);
        for (std::size_t layer = 0; layer < num_layers; ++layer)
        {
            const std::string prefix = block_prefix(layer);

            RmsNorm attn_norm(hidden_dim, rms_eps, required_tensor(reader, prefix + "attn_norm.weight", {hidden_dim}));

            MultiHeadAttention attention(
                hidden_dim, num_heads, /*causal=*/true, rope_theta, max_position_embeddings,
                required_tensor(reader, prefix + "attn_q.weight", {hidden_dim, hidden_dim}),
                optional_tensor(reader, prefix + "attn_q.bias", {hidden_dim}),
                required_tensor(reader, prefix + "attn_k.weight", {hidden_dim, hidden_dim}),
                optional_tensor(reader, prefix + "attn_k.bias", {hidden_dim}),
                required_tensor(reader, prefix + "attn_v.weight", {hidden_dim, hidden_dim}),
                optional_tensor(reader, prefix + "attn_v.bias", {hidden_dim}),
                required_tensor(reader, prefix + "attn_output.weight", {hidden_dim, hidden_dim}),
                optional_tensor(reader, prefix + "attn_output.bias", {hidden_dim}));

            RmsNorm ffn_norm(hidden_dim, rms_eps, required_tensor(reader, prefix + "ffn_norm.weight", {hidden_dim}));

            SwiGLU ffn(
                hidden_dim, intermediate_dim,
                required_tensor(reader, prefix + "ffn_gate.weight", {hidden_dim, intermediate_dim}),
                optional_tensor(reader, prefix + "ffn_gate.bias", {intermediate_dim}),
                required_tensor(reader, prefix + "ffn_up.weight", {hidden_dim, intermediate_dim}),
                optional_tensor(reader, prefix + "ffn_up.bias", {intermediate_dim}),
                required_tensor(reader, prefix + "ffn_down.weight", {intermediate_dim, hidden_dim}),
                optional_tensor(reader, prefix + "ffn_down.bias", {hidden_dim}));

            blocks.emplace_back(std::move(attn_norm), std::move(attention), std::move(ffn_norm), std::move(ffn));
        }

        RmsNorm final_norm(hidden_dim, rms_eps, required_tensor(reader, "output_norm.weight", {hidden_dim}));

        // Tied embeddings: small checkpoints often omit output.weight entirely.
        std::vector<float> lm_head_weights = reader.has_tensor("output.weight")
                                                  ? required_tensor(reader, "output.weight", {hidden_dim, vocab_size})
                                                  : token_embedding;
        Linear lm_head(hidden_dim, vocab_size, std::move(lm_head_weights), {});

        return Model(std::move(embedding), std::move(blocks), std::move(final_norm), std::move(lm_head));
    }

} // namespace mini_inference::loader
