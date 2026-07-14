#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "model/model.h"
#include "tokenizer/bpe_tokenizer.h"

namespace mini_inference::loader
{

    struct GgufCheckpoint
    {
        mini_inference::model::Model model;
        mini_inference::tokenizer::BpeTokenizer tokenizer;
        std::optional<std::size_t> bos_token_id;
        std::optional<std::size_t> eos_token_id;
    };

    // Reads a GGUF file end-to-end into a ready-to-use Model + BpeTokenizer.
    // See gguf_model_loader.h / gguf_tokenizer_loader.h for the scope of each half.
    GgufCheckpoint load_gguf_checkpoint(const std::string &path);

} // namespace mini_inference::loader
