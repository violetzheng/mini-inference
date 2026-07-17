#include "layers/attention.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace mini_inference::layers
{

    namespace
    {

        std::size_t compute_head_dim(std::size_t hidden_dim, std::size_t num_heads)
        {
            if (num_heads == 0)
            {
                throw std::invalid_argument("num_heads must be greater than zero");
            }
            if (hidden_dim == 0 || hidden_dim % num_heads != 0)
            {
                throw std::invalid_argument("hidden_dim must be a positive multiple of num_heads");
            }
            return hidden_dim / num_heads;
        }

        // Pulls out the columns [head_idx * head_dim, (head_idx + 1) * head_dim) belonging
        // to one attention head from a [seq_len, hidden_dim] projection.
        mini_inference::tensor::Tensor extract_head(const mini_inference::tensor::Tensor &full,
                                                      std::size_t head_idx, std::size_t head_dim)
        {
            const std::size_t seq_len = full.shape()[0];
            std::vector<float> values(seq_len * head_dim);
            for (std::size_t row = 0; row < seq_len; ++row)
            {
                for (std::size_t col = 0; col < head_dim; ++col)
                {
                    values[row * head_dim + col] = full.at({row, head_idx * head_dim + col});
                }
            }
            return mini_inference::tensor::Tensor({seq_len, head_dim}, std::move(values));
        }

    } // namespace

    MultiHeadAttention::MultiHeadAttention(std::size_t hidden_dim, std::size_t num_heads, bool causal,
                                            float rope_theta, std::size_t max_position_embeddings,
                                            std::vector<float> q_weights, std::vector<float> q_bias,
                                            std::vector<float> k_weights, std::vector<float> k_bias,
                                            std::vector<float> v_weights, std::vector<float> v_bias,
                                            std::vector<float> o_weights, std::vector<float> o_bias)
        : hidden_dim_(hidden_dim),
          num_heads_(num_heads),
          head_dim_(compute_head_dim(hidden_dim, num_heads)),
          causal_(causal),
          q_proj_(Linear(hidden_dim_, hidden_dim_, std::move(q_weights), std::move(q_bias))),
          k_proj_(Linear(hidden_dim_, hidden_dim_, std::move(k_weights), std::move(k_bias))),
          v_proj_(Linear(hidden_dim_, hidden_dim_, std::move(v_weights), std::move(v_bias))),
          o_proj_(Linear(hidden_dim_, hidden_dim_, std::move(o_weights), std::move(o_bias))),
          rope_(head_dim_, rope_theta, max_position_embeddings),
          softmax_(1)
    {
    }

    MultiHeadAttention::MultiHeadAttention(std::size_t hidden_dim, std::size_t num_heads, bool causal,
                                            float rope_theta, std::size_t max_position_embeddings,
                                            LinearLayer q_proj, LinearLayer k_proj,
                                            LinearLayer v_proj, LinearLayer o_proj)
        : hidden_dim_(hidden_dim),
          num_heads_(num_heads),
          head_dim_(compute_head_dim(hidden_dim, num_heads)),
          causal_(causal),
          q_proj_(std::move(q_proj)),
          k_proj_(std::move(k_proj)),
          v_proj_(std::move(v_proj)),
          o_proj_(std::move(o_proj)),
          rope_(head_dim_, rope_theta, max_position_embeddings),
          softmax_(1)
    {
        if (mini_inference::layers::in_features(q_proj_) != hidden_dim_ ||
            mini_inference::layers::out_features(q_proj_) != hidden_dim_ ||
            mini_inference::layers::in_features(k_proj_) != hidden_dim_ ||
            mini_inference::layers::out_features(k_proj_) != hidden_dim_ ||
            mini_inference::layers::in_features(v_proj_) != hidden_dim_ ||
            mini_inference::layers::out_features(v_proj_) != hidden_dim_ ||
            mini_inference::layers::in_features(o_proj_) != hidden_dim_ ||
            mini_inference::layers::out_features(o_proj_) != hidden_dim_)
        {
            throw std::invalid_argument("attention projection dimensions do not match hidden_dim");
        }
    }

    std::size_t MultiHeadAttention::hidden_dim() const
    {
        return hidden_dim_;
    }

    std::size_t MultiHeadAttention::num_heads() const
    {
        return num_heads_;
    }

    std::size_t MultiHeadAttention::head_dim() const
    {
        return head_dim_;
    }

    bool MultiHeadAttention::causal() const
    {
        return causal_;
    }

    mini_inference::tensor::Tensor MultiHeadAttention::forward(const mini_inference::tensor::Tensor &input,
                                                                 std::size_t position_offset) const
    {
        if (input.rank() != 2)
        {
            throw std::invalid_argument("multi-head attention expects a 2D input tensor");
        }

        const auto &shape = input.shape();
        const std::size_t seq_len = shape[0];
        if (shape[1] != hidden_dim_)
        {
            throw std::invalid_argument("input feature count does not match attention hidden_dim");
        }

        const mini_inference::tensor::Tensor q = mini_inference::layers::forward(q_proj_, input);
        const mini_inference::tensor::Tensor k = mini_inference::layers::forward(k_proj_, input);
        const mini_inference::tensor::Tensor v = mini_inference::layers::forward(v_proj_, input);

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        const float neg_inf = -std::numeric_limits<float>::infinity();

        std::vector<float> merged_values(seq_len * hidden_dim_, 0.0f);

        for (std::size_t head = 0; head < num_heads_; ++head)
        {
            const mini_inference::tensor::Tensor q_head =
                rope_.forward(extract_head(q, head, head_dim_), position_offset);
            const mini_inference::tensor::Tensor k_head =
                rope_.forward(extract_head(k, head, head_dim_), position_offset);
            const mini_inference::tensor::Tensor v_head = extract_head(v, head, head_dim_);

            mini_inference::tensor::Tensor scores({seq_len, seq_len});
            for (std::size_t i = 0; i < seq_len; ++i)
            {
                for (std::size_t j = 0; j < seq_len; ++j)
                {
                    //avoid peaking into future
                    if (causal_ && j > i)
                    {
                        scores.at({i, j}) = neg_inf;
                        continue;
                    }

                    float dot = 0.0f;
                    for (std::size_t d = 0; d < head_dim_; ++d)
                    {
                        dot += q_head.at({i, d}) * k_head.at({j, d});
                    }

                    // how much should token i pay attention to token j
                    scores.at({i, j}) = dot * scale;
                }
            }

            const mini_inference::tensor::Tensor attn = softmax_.forward(scores);

            for (std::size_t i = 0; i < seq_len; ++i)
            {
                for (std::size_t d = 0; d < head_dim_; ++d)
                {
                    float sum = 0.0f;
                    for (std::size_t j = 0; j < seq_len; ++j)
                    {
                        sum += attn.at({i, j}) * v_head.at({j, d});
                    }
                    merged_values[i * hidden_dim_ + head * head_dim_ + d] = sum;
                }
            }
        }

        const mini_inference::tensor::Tensor merged({seq_len, hidden_dim_}, std::move(merged_values));
        return mini_inference::layers::forward(o_proj_, merged);
    }

    mini_inference::tensor::Tensor MultiHeadAttention::forward(const mini_inference::tensor::Tensor &input,
                                                                 KvCache &cache) const
    {
        if (input.rank() != 2)
        {
            throw std::invalid_argument("multi-head attention expects a 2D input tensor");
        }

        const auto &shape = input.shape();
        const std::size_t num_new = shape[0];
        if (shape[1] != hidden_dim_)
        {
            throw std::invalid_argument("input feature count does not match attention hidden_dim");
        }
        if (cache.hidden_dim() != hidden_dim_)
        {
            throw std::invalid_argument("kv cache hidden_dim does not match attention hidden_dim");
        }

        const std::size_t position_offset = cache.length();

        const mini_inference::tensor::Tensor q = mini_inference::layers::forward(q_proj_, input);
        const mini_inference::tensor::Tensor k = mini_inference::layers::forward(k_proj_, input);
        const mini_inference::tensor::Tensor v = mini_inference::layers::forward(v_proj_, input);

        // Rotate Q per head up front (kept for the score loop below) and assemble the
        // rotated K into a full-width buffer so it can be appended to the cache in one shot.
        std::vector<mini_inference::tensor::Tensor> rotated_q_heads;
        rotated_q_heads.reserve(num_heads_);
        std::vector<float> rotated_k_values(num_new * hidden_dim_);

        for (std::size_t head = 0; head < num_heads_; ++head)
        {
            rotated_q_heads.push_back(rope_.forward(extract_head(q, head, head_dim_), position_offset));
            const mini_inference::tensor::Tensor k_head_rotated =
                rope_.forward(extract_head(k, head, head_dim_), position_offset);
            for (std::size_t row = 0; row < num_new; ++row)
            {
                for (std::size_t col = 0; col < head_dim_; ++col)
                {
                    rotated_k_values[row * hidden_dim_ + head * head_dim_ + col] =
                        k_head_rotated.at({row, col});
                }
            }
        }

        const mini_inference::tensor::Tensor rotated_k({num_new, hidden_dim_}, std::move(rotated_k_values));
        cache.append(rotated_k, v);

        const mini_inference::tensor::Tensor cached_k = cache.keys();
        const mini_inference::tensor::Tensor cached_v = cache.values();
        const std::size_t total_len = cache.length();

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        const float neg_inf = -std::numeric_limits<float>::infinity();

        std::vector<float> merged_values(num_new * hidden_dim_, 0.0f);

        for (std::size_t head = 0; head < num_heads_; ++head)
        {
            const mini_inference::tensor::Tensor &q_head = rotated_q_heads[head];
            const mini_inference::tensor::Tensor k_head = extract_head(cached_k, head, head_dim_);
            const mini_inference::tensor::Tensor v_head = extract_head(cached_v, head, head_dim_);

            mini_inference::tensor::Tensor scores({num_new, total_len});
            for (std::size_t i = 0; i < num_new; ++i)
            {
                for (std::size_t j = 0; j < total_len; ++j)
                {
                    //avoid peaking into future
                    if (causal_ && j > position_offset + i)
                    {
                        scores.at({i, j}) = neg_inf;
                        continue;
                    }

                    float dot = 0.0f;
                    for (std::size_t d = 0; d < head_dim_; ++d)
                    {
                        dot += q_head.at({i, d}) * k_head.at({j, d});
                    }

                    scores.at({i, j}) = dot * scale;
                }
            }

            const mini_inference::tensor::Tensor attn = softmax_.forward(scores);

            for (std::size_t i = 0; i < num_new; ++i)
            {
                for (std::size_t d = 0; d < head_dim_; ++d)
                {
                    float sum = 0.0f;
                    for (std::size_t j = 0; j < total_len; ++j)
                    {
                        sum += attn.at({i, j}) * v_head.at({j, d});
                    }
                    merged_values[i * hidden_dim_ + head * head_dim_ + d] = sum;
                }
            }
        }

        const mini_inference::tensor::Tensor merged({num_new, hidden_dim_}, std::move(merged_values));
        return mini_inference::layers::forward(o_proj_, merged);
    }

} // namespace mini_inference::layers
