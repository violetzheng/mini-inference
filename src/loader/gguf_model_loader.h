#pragma once

#include "loader/gguf_reader.h"
#include "model/model.h"

namespace mini_inference::loader
{

    // Builds a Model from a GGUF file's tensors and "llama.*" metadata. Only
    // general.architecture == "llama" with head_count_kv == head_count is supported
    // (MultiHeadAttention has no grouped-query-attention support). Throws
    // invalid_argument or out_of_range on anything this loader can't represent.
    mini_inference::model::Model build_model(const GgufReader &reader);

} // namespace mini_inference::loader
