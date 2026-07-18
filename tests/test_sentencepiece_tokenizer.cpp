#include "tokenizer/sentencepiece_tokenizer.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using mini_inference::tokenizer::SentencePieceTokenizer;

namespace
{

    int failures = 0;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            std::cerr << "FAILED: " << message << std::endl;
            ++failures;
        }
    }

    // Small hand-built vocabulary exercising: multi-step merging ("hi" -> "▁hi"),
    // score-priority between two valid merges at the same round ("de" over "cd"),
    // and byte-fallback for a character with no vocab piece of its own ('z').
    //
    // ids:  0="▁"  1="h"  2="i"  3="a"  4="c"  5="d"  6="e"
    //       7="▁h"(1.0)  8="▁ha"(5.0)  9="hi"(2.0)  10="▁hi"(3.0)
    //       11="cd"(1.0)  12="de"(5.0)  13="<0x7A>"(0.0, byte-fallback for 'z')
    SentencePieceTokenizer make_tokenizer()
    {
        std::vector<std::string> pieces = {
            "\xE2\x96\x81", "h", "i", "a", "c", "d", "e",
            "\xE2\x96\x81h", "\xE2\x96\x81ha", "hi", "\xE2\x96\x81hi",
            "cd", "de", "<0x7A>",
        };
        std::vector<float> scores = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 5.0f, 2.0f, 3.0f,
            1.0f, 5.0f, 0.0f,
        };
        return SentencePieceTokenizer::from_vocab(std::move(pieces), std::move(scores));
    }

    void test_multi_step_merge_and_round_trip()
    {
        const SentencePieceTokenizer tokenizer = make_tokenizer();

        // "hi" -> normalize "▁hi" -> merges "h"+"i"->"hi" (score 2.0, beats "▁"+"h"->"▁h"
        // at score 1.0) -> then "▁"+"hi"->"▁hi" (score 3.0) -> single token id 10.
        const std::vector<std::size_t> ids = tokenizer.encode("hi");
        expect(ids.size() == 1 && ids[0] == 10, "encode(\"hi\") merges down to the single '\xE2\x96\x81hi' token");
        expect(tokenizer.decode(ids) == "hi", "decode(encode(\"hi\")) round-trips");
    }

    void test_merge_requires_multiple_rounds()
    {
        const SentencePieceTokenizer tokenizer = make_tokenizer();

        // "ha" -> normalize "▁ha" -> round 1 only "▁"+"h"->"▁h" is a valid pair (score
        // 1.0; "h"+"a" isn't a vocab piece) -> round 2 "▁h"+"a"->"▁ha" (score 5.0).
        const std::vector<std::size_t> ids = tokenizer.encode("ha");
        expect(ids.size() == 1 && ids[0] == 8, "encode(\"ha\") merges across two rounds to '\xE2\x96\x81ha'");
        expect(tokenizer.decode(ids) == "ha", "decode(encode(\"ha\")) round-trips");
    }

    void test_score_breaks_ties_between_competing_merges()
    {
        const SentencePieceTokenizer tokenizer = make_tokenizer();

        // "cde" -> normalize "▁cde" -> round 1 both "c"+"d"->"cd" (score 1.0) and
        // "d"+"e"->"de" (score 5.0) are valid; the higher-scoring "de" must win, leaving
        // "c" unmerged (no "▁c" or "cde" piece exists to merge further).
        const std::vector<std::size_t> ids = tokenizer.encode("cde");
        expect(ids.size() == 3 && ids[0] == 0 && ids[1] == 4 && ids[2] == 12,
               "encode(\"cde\") prefers the higher-scoring 'de' merge over 'cd'");
    }

    void test_byte_fallback_for_unknown_character()
    {
        const SentencePieceTokenizer tokenizer = make_tokenizer();

        // 'z' has no vocab piece of its own and no merge partner, so it falls back to
        // its raw byte via the "<0x7A>" piece.
        const std::vector<std::size_t> ids = tokenizer.encode("z");
        expect(ids.size() == 2 && ids[0] == 0 && ids[1] == 13, "encode(\"z\") falls back to its raw byte token");
        expect(tokenizer.decode(ids) == "z", "decode(encode(\"z\")) round-trips through byte-fallback");
    }

    void test_accessors()
    {
        const SentencePieceTokenizer tokenizer = make_tokenizer();
        expect(tokenizer.vocab_size() == 14, "vocab_size reflects the constructed vocab");
        expect(tokenizer.token_bytes(0) == "\xE2\x96\x81", "token_bytes(0) is the meta-space piece");
        expect(tokenizer.token_bytes(13) == "<0x7A>", "token_bytes(13) is the byte-fallback piece");
    }

    void test_from_vocab_validation()
    {
        bool threw_empty = false;
        try
        {
            (void)SentencePieceTokenizer::from_vocab({}, {});
        }
        catch (const std::invalid_argument &)
        {
            threw_empty = true;
        }
        expect(threw_empty, "empty vocabulary throws");

        bool threw_mismatch = false;
        try
        {
            (void)SentencePieceTokenizer::from_vocab({"a", "b"}, {0.0f});
        }
        catch (const std::invalid_argument &)
        {
            threw_mismatch = true;
        }
        expect(threw_mismatch, "mismatched pieces/scores lengths throws");
    }

} // namespace

int main()
{
    test_multi_step_merge_and_round_trip();
    test_merge_requires_multiple_rounds();
    test_score_breaks_ties_between_competing_merges();
    test_byte_fallback_for_unknown_character();
    test_accessors();
    test_from_vocab_validation();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All sentencepiece tokenizer tests passed" << std::endl;
    return 0;
}
