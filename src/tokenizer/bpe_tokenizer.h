#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mini_inference::tokenizer
{

    // Byte-level BPE (Sennrich et al.), operating directly on raw bytes with no
    // word/whitespace pre-splitting. Base vocabulary is the 256 byte values, so any
    // input is always tokenizable. A BpeTokenizer is only produced by train() or
    // from_vocab(): the learned vocab and merge table are always internally consistent.
    class BpeTokenizer
    {
    public:
        // left_id + right_id -> new_token_id; rank is the entry's index in from_vocab()'s vector.
        struct VocabMerge
        {
            std::size_t left_id;
            std::size_t right_id;
            std::size_t new_token_id;
        };

        static BpeTokenizer train(const std::string &corpus, std::size_t num_merges);

        // Builds a tokenizer from an already-resolved vocab + merge table (e.g. a loaded
        // file format's vocab) instead of training one. id_to_bytes[i] is token i's raw bytes.
        static BpeTokenizer from_vocab(std::vector<std::string> id_to_bytes,
                                       std::vector<VocabMerge> merges);

        std::vector<std::size_t> encode(const std::string &text) const;
        std::string decode(const std::vector<std::size_t> &token_ids) const;

        std::size_t vocab_size() const;
        const std::string &token_bytes(std::size_t token_id) const;

    private:
        struct MergeRule
        {
            std::size_t rank;
            std::size_t new_token_id;
        };

        struct PairHash
        {
            std::size_t operator()(const std::pair<std::size_t, std::size_t> &pair) const;
        };

        BpeTokenizer() = default;

        std::size_t apply_merge(std::size_t left_id, std::size_t right_id, std::size_t rank);

        std::vector<std::string> id_to_bytes_{};
        std::unordered_map<std::pair<std::size_t, std::size_t>, MergeRule, PairHash> merges_{};
    };

} // namespace mini_inference::tokenizer
