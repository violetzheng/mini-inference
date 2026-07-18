#include "tokenizer/sentencepiece_tokenizer.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace mini_inference::tokenizer
{

    namespace
    {

        // SentencePiece's meta-space codepoint (U+2581 '▁'), representing a literal
        // space in piece text.
        const std::string kMetaSpace = "\xE2\x96\x81";

        std::size_t utf8_char_length(unsigned char lead_byte)
        {
            if ((lead_byte & 0x80) == 0x00)
            {
                return 1;
            }
            if ((lead_byte & 0xE0) == 0xC0)
            {
                return 2;
            }
            if ((lead_byte & 0xF0) == 0xE0)
            {
                return 3;
            }
            if ((lead_byte & 0xF8) == 0xF0)
            {
                return 4;
            }
            // Not a valid UTF-8 lead byte; treat as a single raw byte so byte-fallback
            // can still round-trip it.
            return 1;
        }

        // Replaces every literal space with the meta-space codepoint and prepends one,
        // matching SentencePiece's convention that encode() always starts "as if" the
        // text were preceded by a word boundary.
        std::string normalize(const std::string &text)
        {
            std::string normalized;
            normalized.reserve(text.size() + kMetaSpace.size());
            for (char c : text)
            {
                if (c == ' ')
                {
                    normalized += kMetaSpace;
                }
                else
                {
                    normalized += c;
                }
            }
            if (normalized.compare(0, kMetaSpace.size(), kMetaSpace) != 0)
            {
                normalized = kMetaSpace + normalized;
            }
            return normalized;
        }

        std::vector<std::string> split_utf8_codepoints(const std::string &text)
        {
            std::vector<std::string> symbols;
            std::size_t i = 0;
            while (i < text.size())
            {
                std::size_t len = utf8_char_length(static_cast<unsigned char>(text[i]));
                len = std::min(len, text.size() - i);
                symbols.push_back(text.substr(i, len));
                i += len;
            }
            return symbols;
        }

        std::string byte_fallback_piece(unsigned char byte)
        {
            static const char *hex_digits = "0123456789ABCDEF";
            std::string piece = "<0x00>";
            piece[3] = hex_digits[(byte >> 4) & 0xF];
            piece[4] = hex_digits[byte & 0xF];
            return piece;
        }

        int hex_value(char c)
        {
            if (c >= '0' && c <= '9')
            {
                return c - '0';
            }
            if (c >= 'A' && c <= 'F')
            {
                return c - 'A' + 10;
            }
            return -1;
        }

        // Parses a "<0xXX>" byte-fallback piece back to its raw byte; returns false
        // (leaving out_byte unset) for any piece that isn't in that exact form.
        bool parse_byte_fallback_piece(const std::string &piece, unsigned char &out_byte)
        {
            if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>')
            {
                return false;
            }
            const int hi = hex_value(piece[3]);
            const int lo = hex_value(piece[4]);
            if (hi < 0 || lo < 0)
            {
                return false;
            }
            out_byte = static_cast<unsigned char>((hi << 4) | lo);
            return true;
        }

    } // namespace

    SentencePieceTokenizer SentencePieceTokenizer::from_vocab(std::vector<std::string> pieces,
                                                               std::vector<float> scores)
    {
        if (pieces.empty())
        {
            throw std::invalid_argument("sentencepiece vocabulary must not be empty");
        }
        if (pieces.size() != scores.size())
        {
            throw std::invalid_argument("sentencepiece pieces and scores must have the same length");
        }

        SentencePieceTokenizer tokenizer;
        tokenizer.id_to_piece_ = std::move(pieces);
        tokenizer.scores_ = std::move(scores);

        tokenizer.piece_to_id_.reserve(tokenizer.id_to_piece_.size());
        for (std::size_t id = 0; id < tokenizer.id_to_piece_.size(); ++id)
        {
            // First occurrence wins; from_vocab is a thin loader-facing constructor,
            // not a strict vocab validator.
            tokenizer.piece_to_id_.emplace(tokenizer.id_to_piece_[id], id);
        }

        return tokenizer;
    }

    std::vector<std::size_t> SentencePieceTokenizer::encode(const std::string &text) const
    {
        std::vector<std::string> symbols = split_utf8_codepoints(normalize(text));

        // Repeatedly merge the single highest-scoring adjacent pair whose concatenation
        // is a known vocab piece, until no such pair remains. Same "rescan everything,
        // apply the best merge, repeat" shape as BpeTokenizer::encode, just ranked by
        // score instead of a precomputed merge-rank table.
        while (symbols.size() > 1)
        {
            float best_score = -std::numeric_limits<float>::infinity();
            std::size_t best_index = 0;
            bool found = false;

            for (std::size_t i = 0; i + 1 < symbols.size(); ++i)
            {
                const auto it = piece_to_id_.find(symbols[i] + symbols[i + 1]);
                if (it == piece_to_id_.end())
                {
                    continue;
                }
                const float score = scores_[it->second];
                if (!found || score > best_score)
                {
                    best_score = score;
                    best_index = i;
                    found = true;
                }
            }

            if (!found)
            {
                break;
            }

            symbols[best_index] += symbols[best_index + 1];
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index) + 1);
        }

        std::vector<std::size_t> ids;
        ids.reserve(symbols.size());
        for (const std::string &symbol : symbols)
        {
            const auto it = piece_to_id_.find(symbol);
            if (it != piece_to_id_.end())
            {
                ids.push_back(it->second);
                continue;
            }

            // Byte fallback: no single piece covers this symbol (e.g. an unseen
            // character), so encode it byte-by-byte instead.
            for (unsigned char byte : symbol)
            {
                const auto byte_it = piece_to_id_.find(byte_fallback_piece(byte));
                if (byte_it == piece_to_id_.end())
                {
                    throw std::invalid_argument(
                        "sentencepiece vocabulary has no piece or byte-fallback token for an input character");
                }
                ids.push_back(byte_it->second);
            }
        }

        return ids;
    }

    std::string SentencePieceTokenizer::decode(const std::vector<std::size_t> &token_ids) const
    {
        std::string text;
        for (std::size_t id : token_ids)
        {
            assert(id < id_to_piece_.size());
            const std::string &piece = id_to_piece_[id];

            unsigned char byte = 0;
            if (parse_byte_fallback_piece(piece, byte))
            {
                text += static_cast<char>(byte);
            }
            else
            {
                text += piece;
            }
        }

        std::string result;
        result.reserve(text.size());
        for (std::size_t i = 0; i < text.size();)
        {
            if (text.compare(i, kMetaSpace.size(), kMetaSpace) == 0)
            {
                result += ' ';
                i += kMetaSpace.size();
            }
            else
            {
                result += text[i];
                ++i;
            }
        }

        // encode() always prepends a meta-space marker to represent "start of text";
        // undo that here rather than leaving a spurious leading space.
        if (!result.empty() && result.front() == ' ')
        {
            result.erase(0, 1);
        }
        return result;
    }

    std::size_t SentencePieceTokenizer::vocab_size() const
    {
        return id_to_piece_.size();
    }

    const std::string &SentencePieceTokenizer::token_bytes(std::size_t token_id) const
    {
        assert(token_id < id_to_piece_.size());
        return id_to_piece_[token_id];
    }

} // namespace mini_inference::tokenizer
