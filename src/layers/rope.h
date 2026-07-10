#pragma once

#include <cstddef>
#include <vector>

#include "tensor/tensor.h"

namespace mini_inference::layers
{

    // Rotary position embedding (RoPE): rotates consecutive feature pairs by an
    // angle proportional to sequence position, injecting position information
    // directly into query/key vectors instead of via additive embeddings.
    class RoPE
    {
    public:
        explicit RoPE(std::size_t dim, float theta = 10000.0f, std::size_t max_position_embeddings = 2048);

        std::size_t dim() const;
        float theta() const;
        std::size_t max_position_embeddings() const;

        // Rotates `input` (shape [seq_len, dim]), treating row i as position (position_offset + i).
        // position_offset lets a caller resume rotation mid-sequence, e.g. when decoding a new
        // token against a KV cache that already holds earlier positions.
        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input,
                                                std::size_t position_offset = 0) const;

        // Rotates `input` using an explicit position id per row, for batched decoding where rows
        // do not correspond to a contiguous position range.
        mini_inference::tensor::Tensor forward(const mini_inference::tensor::Tensor &input,
                                                const std::vector<std::size_t> &positions) const;

    private:
        std::size_t dim_{0};
        float theta_{10000.0f};
        std::size_t max_position_embeddings_{0};
        std::vector<float> inv_freq_{};
        std::vector<float> cos_cache_{};
        std::vector<float> sin_cache_{};

        void build_cache();
    };

} // namespace mini_inference::layers
