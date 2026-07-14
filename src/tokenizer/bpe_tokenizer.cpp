#include "tokenizer/bpe_tokenizer.h"

#include <cassert>
#include <limits>
#include <stdexcept>

namespace mini_inference::tokenizer
{

    namespace
    {

        //letter -> token id
        std::vector<std::size_t> bytes_to_ids(const std::string &text)
        {
            std::vector<std::size_t> ids;
            ids.reserve(text.size());
            for (unsigned char byte : text)
            {
                ids.push_back(static_cast<std::size_t>(byte));
            }
            return ids;
        }

        // Merges every non-overlapping occurrence of (left, right) into new_id 
        std::vector<std::size_t> merge_pair(const std::vector<std::size_t> &sequence,
                                             std::size_t left, std::size_t right, std::size_t new_id)
        {
            std::vector<std::size_t> merged;
            merged.reserve(sequence.size());
            for (std::size_t i = 0; i < sequence.size();)
            {
                if (i + 1 < sequence.size() && sequence[i] == left && sequence[i + 1] == right)
                {
                    merged.push_back(new_id);
                    i += 2;
                }
                else
                {
                    merged.push_back(sequence[i]);
                    ++i;
                }
            }
            return merged;
        }

    } // namespace

    std::size_t BpeTokenizer::PairHash::operator()(const std::pair<std::size_t, std::size_t> &pair) const
    {
        return std::hash<std::size_t>()(pair.first) ^ (std::hash<std::size_t>()(pair.second) << 1);
    }

    std::size_t BpeTokenizer::apply_merge(std::size_t left_id, std::size_t right_id, std::size_t rank)
    {
        const std::size_t new_id = id_to_bytes_.size();
        id_to_bytes_.push_back(id_to_bytes_[left_id] + id_to_bytes_[right_id]);
        merges_[{left_id, right_id}] = MergeRule{rank, new_id};
        return new_id;
    }

    BpeTokenizer BpeTokenizer::train(const std::string &corpus, std::size_t num_merges)
    {
        if (corpus.empty())
        {
            throw std::invalid_argument("bpe training corpus must not be empty");
        }

        BpeTokenizer tokenizer;
        tokenizer.id_to_bytes_.reserve(256 + num_merges);
        for (int byte = 0; byte < 256; ++byte)
        {
            tokenizer.id_to_bytes_.push_back(std::string(1, static_cast<char>(byte)));
        }

        std::vector<std::size_t> sequence = bytes_to_ids(corpus);

        for (std::size_t rank = 0; rank < num_merges; ++rank)
        {
            std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t, PairHash> counts;
            for (std::size_t i = 0; i + 1 < sequence.size(); ++i)
            {
                ++counts[{sequence[i], sequence[i + 1]}];
            }

            if (counts.empty())
            {
                break;
            }

            auto best = counts.begin();
            for (auto it = counts.begin(); it != counts.end(); ++it)
            {
                if (it->second > best->second || (it->second == best->second && it->first < best->first))
                {
                    best = it;
                }
            }

            // Merging a pair that only occurs once wastes a vocab slot with no
            // compression benefit, so stop training early once no pair repeats.
            if (best->second < 2)
            {
                break;
            }

            const auto [left, right] = best->first;
            const std::size_t new_id = tokenizer.apply_merge(left, right, rank);
            sequence = merge_pair(sequence, left, right, new_id);
        }

        return tokenizer;
    }

    std::vector<std::size_t> BpeTokenizer::encode(const std::string &text) const
    {
        std::vector<std::size_t> sequence = bytes_to_ids(text);

        while (sequence.size() > 1)
        {
            std::size_t best_rank = std::numeric_limits<std::size_t>::max();
            std::size_t best_left = 0;
            std::size_t best_right = 0;
            std::size_t best_new_id = 0;
            bool found = false;

            for (std::size_t i = 0; i + 1 < sequence.size(); ++i)
            {
                const auto it = merges_.find({sequence[i], sequence[i + 1]});
                if (it != merges_.end() && it->second.rank < best_rank)
                {
                    best_rank = it->second.rank;
                    best_left = sequence[i];
                    best_right = sequence[i + 1];
                    best_new_id = it->second.new_token_id;
                    found = true;
                }
            }

            if (!found)
            {
                break;
            }

            sequence = merge_pair(sequence, best_left, best_right, best_new_id);
        }

        return sequence;
    }

    std::string BpeTokenizer::decode(const std::vector<std::size_t> &token_ids) const
    {
        std::string text;
        for (std::size_t id : token_ids)
        {
            assert(id < id_to_bytes_.size());
            text += id_to_bytes_[id];
        }
        return text;
    }

    std::size_t BpeTokenizer::vocab_size() const
    {
        return id_to_bytes_.size();
    }

    const std::string &BpeTokenizer::token_bytes(std::size_t token_id) const
    {
        assert(token_id < id_to_bytes_.size());
        return id_to_bytes_[token_id];
    }

} // namespace mini_inference::tokenizer
