#include "layers/transformer_block.h"

#include <cassert>
#include <stdexcept>
#include <vector>

namespace mini_inference::layers
{

    namespace
    {

        std::size_t validate_consistent_hidden_dim(const RmsNorm &attn_norm, const MultiHeadAttention &attention,
                                                     const RmsNorm &ffn_norm, const SwiGLU &ffn)
        {
            const std::size_t hidden_dim = attention.hidden_dim();
            if (attn_norm.dim() != hidden_dim || ffn_norm.dim() != hidden_dim || ffn.hidden_dim() != hidden_dim)
            {
                throw std::invalid_argument("transformer block sub-layer hidden_dim values do not match");
            }
            return hidden_dim;
        }

        mini_inference::tensor::Tensor add_residual(const mini_inference::tensor::Tensor &a,
                                                      const mini_inference::tensor::Tensor &b)
        {
            assert(a.shape() == b.shape());

            std::vector<float> sum(a.numel());
            const std::vector<float> &a_values = a.values();
            const std::vector<float> &b_values = b.values();
            for (std::size_t i = 0; i < sum.size(); ++i)
            {
                sum[i] = a_values[i] + b_values[i];
            }
            return mini_inference::tensor::Tensor(a.shape(), std::move(sum));
        }

    } // namespace

    TransformerBlock::TransformerBlock(RmsNorm attn_norm, MultiHeadAttention attention,
                                        RmsNorm ffn_norm, SwiGLU ffn)
        : hidden_dim_(validate_consistent_hidden_dim(attn_norm, attention, ffn_norm, ffn)),
          num_heads_(attention.num_heads()),
          intermediate_dim_(ffn.intermediate_dim()),
          attn_norm_(std::move(attn_norm)),
          attention_(std::move(attention)),
          ffn_norm_(std::move(ffn_norm)),
          ffn_(std::move(ffn))
    {
    }

    std::size_t TransformerBlock::hidden_dim() const
    {
        return hidden_dim_;
    }

    std::size_t TransformerBlock::num_heads() const
    {
        return num_heads_;
    }

    std::size_t TransformerBlock::intermediate_dim() const
    {
        return intermediate_dim_;
    }

    mini_inference::tensor::Tensor TransformerBlock::forward(const mini_inference::tensor::Tensor &input,
                                                               std::size_t position_offset) const
    {
        if (input.rank() != 2)
        {
            throw std::invalid_argument("transformer block expects a 2D input tensor");
        }

        if (input.shape()[1] != hidden_dim_)
        {
            throw std::invalid_argument("input feature count does not match transformer block hidden_dim");
        }

        const mini_inference::tensor::Tensor normed1 = attn_norm_.forward(input);
        const mini_inference::tensor::Tensor attn_out = attention_.forward(normed1, position_offset);
        const mini_inference::tensor::Tensor residual1 = add_residual(input, attn_out);

        const mini_inference::tensor::Tensor normed2 = ffn_norm_.forward(residual1);
        const mini_inference::tensor::Tensor ffn_out = ffn_.forward(normed2);
        return add_residual(residual1, ffn_out);
    }

    mini_inference::tensor::Tensor TransformerBlock::forward(const mini_inference::tensor::Tensor &input,
                                                               KvCache &cache) const
    {
        if (input.rank() != 2)
        {
            throw std::invalid_argument("transformer block expects a 2D input tensor");
        }

        if (input.shape()[1] != hidden_dim_)
        {
            throw std::invalid_argument("input feature count does not match transformer block hidden_dim");
        }

        const mini_inference::tensor::Tensor normed1 = attn_norm_.forward(input);
        const mini_inference::tensor::Tensor attn_out = attention_.forward(normed1, cache);
        const mini_inference::tensor::Tensor residual1 = add_residual(input, attn_out);

        const mini_inference::tensor::Tensor normed2 = ffn_norm_.forward(residual1);
        const mini_inference::tensor::Tensor ffn_out = ffn_.forward(normed2);
        return add_residual(residual1, ffn_out);
    }

} // namespace mini_inference::layers
