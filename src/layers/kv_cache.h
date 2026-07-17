#pragma once

#include <cstddef>
#include <vector>

#include "tensor/tensor.h"

namespace mini_inference::layers
{

    // Per-attention-layer cache of past keys and values, stored at full hidden
    // width (all heads concatenated, matching the layout k_proj_/v_proj_ produce).
    // A caller appends the newly computed rows for each decode step and reads back
    // every position seen so far, avoiding recomputation of past keys/values.
    class KvCache
    {
    public:
        KvCache(std::size_t max_seq_len, std::size_t hidden_dim);

        std::size_t length() const;
        std::size_t max_seq_len() const;
        std::size_t hidden_dim() const;

        // Appends new rows (shape [n, hidden_dim]) at the current length. Throws
        // std::out_of_range if this would exceed max_seq_len.
        void append(const mini_inference::tensor::Tensor &k, const mini_inference::tensor::Tensor &v);

        mini_inference::tensor::Tensor keys() const;
        mini_inference::tensor::Tensor values() const;

        void reset();

    private:
        std::size_t max_seq_len_{0};
        std::size_t hidden_dim_{0};
        std::size_t length_{0};

        std::vector<float> k_storage_{};
        std::vector<float> v_storage_{};
    };

} // namespace mini_inference::layers
