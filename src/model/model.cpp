#include "model/model.h"

#include <cassert>
#include <stdexcept>

namespace mini_inference::model
{

    namespace
    {

        std::size_t argmax_last_row(const mini_inference::tensor::Tensor &logits)
        {
            assert(logits.rank() == 2);

            const std::size_t last_row = logits.shape()[0] - 1;
            const std::size_t vocab_size = logits.shape()[1];

            std::size_t best_index = 0;
            float best_value = logits.at({last_row, 0});
            for (std::size_t v = 1; v < vocab_size; ++v)
            {
                const float value = logits.at({last_row, v});
                if (value > best_value)
                {
                    best_value = value;
                    best_index = v;
                }
            }
            return best_index;
        }

        std::size_t validate_and_derive_hidden_dim(const mini_inference::layers::Embedding &embedding,
                                                     const std::vector<mini_inference::layers::TransformerBlock> &blocks,
                                                     const mini_inference::layers::RmsNorm &final_norm,
                                                     const mini_inference::layers::Linear &lm_head)
        {
            if (blocks.empty())
            {
                throw std::invalid_argument("model must have at least one transformer block");
            }

            const std::size_t hidden_dim = embedding.hidden_dim();
            for (const auto &block : blocks)
            {
                if (block.hidden_dim() != hidden_dim)
                {
                    throw std::invalid_argument("transformer block hidden_dim does not match embedding hidden_dim");
                }
            }

            if (final_norm.dim() != hidden_dim)
            {
                throw std::invalid_argument("final_norm dim does not match model hidden_dim");
            }

            if (lm_head.in_features() != hidden_dim)
            {
                throw std::invalid_argument("lm_head in_features does not match model hidden_dim");
            }

            if (lm_head.out_features() != embedding.vocab_size())
            {
                throw std::invalid_argument("lm_head out_features does not match embedding vocab_size");
            }

            return hidden_dim;
        }

    } // namespace

    Model::Model(mini_inference::layers::Embedding embedding,
                 std::vector<mini_inference::layers::TransformerBlock> blocks,
                 mini_inference::layers::RmsNorm final_norm,
                 mini_inference::layers::Linear lm_head)
        : vocab_size_(embedding.vocab_size()),
          hidden_dim_(validate_and_derive_hidden_dim(embedding, blocks, final_norm, lm_head)),
          embedding_(std::move(embedding)),
          blocks_(std::move(blocks)),
          final_norm_(std::move(final_norm)),
          lm_head_(std::move(lm_head))
    {
    }

    std::size_t Model::vocab_size() const
    {
        return vocab_size_;
    }

    std::size_t Model::hidden_dim() const
    {
        return hidden_dim_;
    }

    std::size_t Model::num_layers() const
    {
        return blocks_.size();
    }

    mini_inference::tensor::Tensor Model::forward(const std::vector<std::size_t> &token_ids,
                                                    std::size_t position_offset) const
    {
        if (token_ids.empty())
        {
            throw std::invalid_argument("model forward requires at least one token id");
        }

        mini_inference::tensor::Tensor x = embedding_.forward(token_ids);
        for (const auto &block : blocks_)
        {
            x = block.forward(x, position_offset);
        }

        x = final_norm_.forward(x);
        return lm_head_.forward(x);
    }

    std::vector<std::size_t> Model::generate(std::vector<std::size_t> prompt_token_ids,
                                              std::size_t max_new_tokens,
                                              std::optional<std::size_t> eos_token_id) const
    {
        if (prompt_token_ids.empty())
        {
            throw std::invalid_argument("generate requires a non-empty prompt");
        }

        std::vector<std::size_t> token_ids = std::move(prompt_token_ids);
        for (std::size_t step = 0; step < max_new_tokens; ++step)
        {
            const mini_inference::tensor::Tensor logits = forward(token_ids);
            const std::size_t next_token = argmax_last_row(logits);
            token_ids.push_back(next_token);

            if (eos_token_id.has_value() && next_token == *eos_token_id)
            {
                break;
            }
        }
        return token_ids;
    }

} // namespace mini_inference::model
