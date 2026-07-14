#include "loader/gguf_tokenizer_loader.h"
#include "loader/gpt2_byte_encoding.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mini_inference::loader
{

    using mini_inference::tokenizer::BpeTokenizer;

    BpeTokenizer build_tokenizer(const GgufReader &reader)
    {
        const std::string tokenizer_model = reader.metadata_string("tokenizer.ggml.model");
        if (tokenizer_model != "gpt2")
        {
            throw std::invalid_argument("unsupported GGUF tokenizer model '" + tokenizer_model +
                                         "' (only 'gpt2' byte-level BPE is supported)");
        }

        // Vocab/merge entries are GPT-2 display strings (e.g. a space renders as 'Ġ'),
        // resolved by display string first, then decoded to raw bytes below.
        const std::vector<std::string> display_tokens = reader.metadata_string_array("tokenizer.ggml.tokens");

        std::unordered_map<std::string, std::size_t> id_by_display_string;
        id_by_display_string.reserve(display_tokens.size());
        for (std::size_t id = 0; id < display_tokens.size(); ++id)
        {
            id_by_display_string[display_tokens[id]] = id;
        }

        const Gpt2ByteEncoding byte_encoding;
        std::vector<std::string> id_to_bytes;
        id_to_bytes.reserve(display_tokens.size());
        for (const std::string &display_token : display_tokens)
        {
            id_to_bytes.push_back(byte_encoding.decode(display_token));
        }

        std::vector<BpeTokenizer::VocabMerge> merges;
        if (reader.has_metadata("tokenizer.ggml.merges"))
        {
            const std::vector<std::string> merge_lines = reader.metadata_string_array("tokenizer.ggml.merges");
            merges.reserve(merge_lines.size());

            for (const std::string &line : merge_lines)
            {
                const std::size_t space_pos = line.find(' ');
                if (space_pos == std::string::npos)
                {
                    throw std::invalid_argument("malformed GGUF BPE merge entry '" + line +
                                                 "' (expected '<left> <right>')");
                }

                const std::string left = line.substr(0, space_pos);
                const std::string right = line.substr(space_pos + 1);
                const std::string merged = left + right;

                const auto left_it = id_by_display_string.find(left);
                const auto right_it = id_by_display_string.find(right);
                const auto merged_it = id_by_display_string.find(merged);
                if (left_it == id_by_display_string.end() || right_it == id_by_display_string.end() ||
                    merged_it == id_by_display_string.end())
                {
                    throw std::invalid_argument("GGUF BPE merge entry '" + line +
                                                 "' references a token not in the vocabulary");
                }

                merges.push_back(BpeTokenizer::VocabMerge{left_it->second, right_it->second, merged_it->second});
            }
        }

        return BpeTokenizer::from_vocab(std::move(id_to_bytes), std::move(merges));
    }

} // namespace mini_inference::loader
