#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "layers/embedding.h"
#include "layers/kv_cache.h"
#include "layers/linear.h"
#include "layers/linear_layer.h"
#include "layers/rms_norm.h"
#include "layers/transformer_block.h"
#include "tensor/tensor.h"

namespace mini_inference::model
{

    // Full assembled decoder-only transformer: token embedding -> N stacked
    // TransformerBlocks -> final RmsNorm -> lm_head projection -> logits.
    // Composes already-configured sub-layers (mirrors TransformerBlock's own
    // composition idiom one level up). Does not perform checkpoint loading or
    // tokenization - those are separate future submodules.
    class Model
    {
    public:
        Model(mini_inference::layers::Embedding embedding,
              std::vector<mini_inference::layers::TransformerBlock> blocks,
              mini_inference::layers::RmsNorm final_norm,
              mini_inference::layers::LinearLayer lm_head);

        std::size_t vocab_size() const;
        std::size_t hidden_dim() const;
        std::size_t num_layers() const;

        mini_inference::tensor::Tensor forward(const std::vector<std::size_t> &token_ids,
                                                std::size_t position_offset = 0) const;

        // Cache-aware path: one mini_inference::layers::KvCache per transformer block,
        // in block order. Each block reads/appends its own caches[i].
        mini_inference::tensor::Tensor forward(const std::vector<std::size_t> &token_ids,
                                                std::vector<mini_inference::layers::KvCache> &caches) const;

        // Builds one fresh KvCache per transformer block, each sized for up to
        // max_seq_len positions.
        std::vector<mini_inference::layers::KvCache> make_kv_cache(std::size_t max_seq_len) const;

        // Greedy autoregressive decoding: repeatedly picks the argmax next token and
        // appends it, until max_new_tokens are generated or eos_token_id is produced.
        // Prefills a KV cache with the whole prompt in one forward pass, then decodes
        // one new token at a time against the cache instead of re-running the full
        // growing sequence at every step.
        std::vector<std::size_t> generate(std::vector<std::size_t> prompt_token_ids,
                                           std::size_t max_new_tokens,
                                           std::optional<std::size_t> eos_token_id = std::nullopt) const;

    private:
        std::size_t vocab_size_{0};
        std::size_t hidden_dim_{0};

        mini_inference::layers::Embedding embedding_;
        std::vector<mini_inference::layers::TransformerBlock> blocks_;
        mini_inference::layers::RmsNorm final_norm_;
        mini_inference::layers::LinearLayer lm_head_;
    };

} // namespace mini_inference::model
