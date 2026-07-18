#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace mini_inference::tokenizer
{

    // Score-ranked BPE tokenizer matching SentencePiece's "llama" GGUF vocabulary
    // convention: pieces carry a float score (rather than an explicit ordered merge
    // list like GPT-2's byte-BPE), a leading/inter-word space is represented as the
    // single codepoint U+2581 '▁', and codepoints missing from the vocabulary fall
    // back to per-byte "<0xXX>" pieces (a convention every SentencePiece GGUF vocab
    // includes). A SentencePieceTokenizer is only produced by from_vocab(): the
    // vocab and its piece->id lookup are always internally consistent.
    class SentencePieceTokenizer
    {
    public:
        // pieces[i]/scores[i] is token i's raw UTF-8 text / merge priority (higher
        // score merges first). pieces.size() must equal scores.size().
        static SentencePieceTokenizer from_vocab(std::vector<std::string> pieces, std::vector<float> scores);

        std::vector<std::size_t> encode(const std::string &text) const;
        std::string decode(const std::vector<std::size_t> &token_ids) const;

        std::size_t vocab_size() const;
        const std::string &token_bytes(std::size_t token_id) const;

    private:
        SentencePieceTokenizer() = default;

        std::vector<std::string> id_to_piece_{};
        std::vector<float> scores_{};
        std::unordered_map<std::string, std::size_t> piece_to_id_{};
    };

} // namespace mini_inference::tokenizer
