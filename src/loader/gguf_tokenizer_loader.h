#pragma once

#include "loader/gguf_reader.h"
#include "tokenizer/bpe_tokenizer.h"

namespace mini_inference::loader
{

    // Builds a BpeTokenizer from a GGUF file's tokenizer.ggml.* metadata. Only
    // tokenizer.ggml.model == "gpt2" (byte-level BPE) is supported; SentencePiece
    // vocabularies use a different algorithm this codebase doesn't implement.
    mini_inference::tokenizer::BpeTokenizer build_tokenizer(const GgufReader &reader);

} // namespace mini_inference::loader
