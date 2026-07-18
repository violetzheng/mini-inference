#pragma once

#include <variant>

#include "loader/gguf_reader.h"
#include "tokenizer/bpe_tokenizer.h"
#include "tokenizer/sentencepiece_tokenizer.h"

namespace mini_inference::loader
{

    // Either tokenizer type a GGUF checkpoint may use, keyed by tokenizer.ggml.model.
    using Tokenizer =
        std::variant<mini_inference::tokenizer::BpeTokenizer, mini_inference::tokenizer::SentencePieceTokenizer>;

    // Builds a tokenizer from a GGUF file's tokenizer.ggml.* metadata, dispatching on
    // tokenizer.ggml.model: "gpt2" builds a byte-level BPE BpeTokenizer, "llama" builds
    // a score-ranked SentencePieceTokenizer. Any other value throws invalid_argument.
    Tokenizer build_tokenizer(const GgufReader &reader);

} // namespace mini_inference::loader
