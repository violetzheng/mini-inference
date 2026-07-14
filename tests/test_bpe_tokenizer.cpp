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

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All bpe_tokenizer tests passed" << std::endl;
    return 0;
}
