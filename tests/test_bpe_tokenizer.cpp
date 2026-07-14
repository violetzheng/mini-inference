#include "tokenizer/bpe_tokenizer.h"

#include <iostream>
#include <stdexcept>
#include <string>

using mini_inference::tokenizer::BpeTokenizer;

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

} // namespace

int main()
{
    // Classic Sennrich et al. textbook example: corpus "aaabdaaabac", 3 merges.
    // Hand-traced merge order: "aa" (count 4) -> Z(256), then "ab" (count 2, beats
    // "Za" on the lexicographic tie-break since (97,98) < (256,97)) -> Y(257), then
    // "ZY" (count 2) -> X(258). Final compressed corpus: "XdXac".
    const std::string corpus = "aaabdaaabac";
    BpeTokenizer tokenizer = BpeTokenizer::train(corpus, 3);

    expect(tokenizer.vocab_size() == 259, "vocab_size == 256 base + 3 merges");
    expect(tokenizer.token_bytes(256) == "aa", "first merge is 'aa'");
    expect(tokenizer.token_bytes(257) == "ab", "second merge is 'ab'");
    expect(tokenizer.token_bytes(258) == "aaab", "third merge is 'aa'+'ab' == 'aaab'");

    const std::vector<std::size_t> expected_encoding = {
        258, static_cast<std::size_t>('d'), 258, static_cast<std::size_t>('a'), static_cast<std::size_t>('c')};
    expect(tokenizer.encode(corpus) == expected_encoding, "encode(corpus) matches hand-traced 'XdXac'");

    expect(tokenizer.decode(expected_encoding) == corpus, "decode(encode(corpus)) round-trips exactly");

    // Round-trip on text unseen during training, including a multi-byte UTF-8
    // character, demonstrates byte-level tokenization never fails on novel input.
    const std::string novel_text = "café, bad";
    expect(tokenizer.decode(tokenizer.encode(novel_text)) == novel_text,
           "decode(encode(text)) round-trips for novel UTF-8 text");

    // num_merges = 0: identity mapping onto the base byte vocabulary.
    BpeTokenizer identity_tokenizer = BpeTokenizer::train(corpus, 0);
    expect(identity_tokenizer.vocab_size() == 256, "vocab_size == 256 with no merges");

    const std::vector<std::size_t> raw_bytes = {
        static_cast<std::size_t>('a'), static_cast<std::size_t>('b'), static_cast<std::size_t>('c')};
    expect(identity_tokenizer.encode("abc") == raw_bytes, "encode with no merges is the identity byte mapping");

    bool threw_empty_corpus = false;
    try
    {
        (void)BpeTokenizer::train("", 3);
    }
    catch (const std::invalid_argument &)
    {
        threw_empty_corpus = true;
    }
    expect(threw_empty_corpus, "empty training corpus throws");

    // encode()'s byte-to-id mapping assumes ids 0-255 are the base byte values in
    // order, so the vocab here must follow that layout too, plus one appended merge.
    std::vector<std::string> base_vocab;
    base_vocab.reserve(257);
    for (int byte = 0; byte < 256; ++byte)
    {
        base_vocab.push_back(std::string(1, static_cast<char>(byte)));
    }
    base_vocab.push_back("ab"); // id 256: merge of 'a' (97) + 'b' (98)

    BpeTokenizer from_vocab_tokenizer = BpeTokenizer::from_vocab(
        base_vocab, {BpeTokenizer::VocabMerge{static_cast<std::size_t>('a'), static_cast<std::size_t>('b'), 256}});

    expect(from_vocab_tokenizer.vocab_size() == 257, "from_vocab vocab_size matches supplied vocab");
    expect(from_vocab_tokenizer.token_bytes(256) == "ab", "from_vocab token_bytes returns the supplied entry");
    expect(from_vocab_tokenizer.encode("ab") == std::vector<std::size_t>{256}, "from_vocab encode applies the supplied merge");
    expect(from_vocab_tokenizer.decode({256}) == "ab", "from_vocab decode round-trips the merged token");

    bool threw_empty_vocab = false;
    try
    {
        (void)BpeTokenizer::from_vocab({}, {});
    }
    catch (const std::invalid_argument &)
    {
        threw_empty_vocab = true;
    }
    expect(threw_empty_vocab, "from_vocab with an empty vocabulary throws");

    bool threw_out_of_range_merge = false;
    try
    {
        (void)BpeTokenizer::from_vocab({"a", "b"}, {BpeTokenizer::VocabMerge{0, 1, 99}});
    }
    catch (const std::invalid_argument &)
    {
        threw_out_of_range_merge = true;
    }
    expect(threw_out_of_range_merge, "from_vocab with an out-of-range merge id throws");

    bool threw_duplicate_merge = false;
    try
    {
        (void)BpeTokenizer::from_vocab({"a", "b", "ab", "aba"},
                                       {BpeTokenizer::VocabMerge{0, 1, 2}, BpeTokenizer::VocabMerge{0, 1, 3}});
    }
    catch (const std::invalid_argument &)
    {
        threw_duplicate_merge = true;
    }
    expect(threw_duplicate_merge, "from_vocab with a duplicate merge pair throws");

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All bpe_tokenizer tests passed" << std::endl;
    return 0;
}
