#include "loader/gguf_tokenizer_loader.h"
#include "gguf_test_helpers.h"
#include "loader/gguf_reader.h"
#include "loader/gpt2_byte_encoding.h"
#include "tokenizer/bpe_tokenizer.h"
#include "tokenizer/sentencepiece_tokenizer.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using mini_inference::loader::GgufReader;
using mini_inference::loader::Gpt2ByteEncoding;
using mini_inference::tests::GgufBufferBuilder;
using mini_inference::tokenizer::BpeTokenizer;
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

} // namespace

int main()
{
    // --- 256 base-byte tokens (ids 0-255, in byte order) plus two learned merges ---
    {
        const Gpt2ByteEncoding encoding;
        std::vector<std::string> tokens;
        tokens.reserve(258);
        for (int byte = 0; byte < 256; ++byte)
        {
            tokens.push_back(encoding.encode(std::string(1, static_cast<char>(byte))));
        }
        tokens.push_back("ab"); // id 256: merge of 'a' (97) + 'b' (98)
        tokens.push_back("Ġa"); // id 257: merge of ' ' (32, displayed as 'Ġ') + 'a' (97)

        GgufBufferBuilder builder;
        builder.add_string_kv("tokenizer.ggml.model", "gpt2");
        builder.add_string_array_kv("tokenizer.ggml.tokens", tokens);
        builder.add_string_array_kv("tokenizer.ggml.merges", {"a b", "Ġ a"});

        GgufReader reader(builder.build());
        mini_inference::loader::Tokenizer loaded = mini_inference::loader::build_tokenizer(reader);
        expect(std::holds_alternative<BpeTokenizer>(loaded), "tokenizer.ggml.model == \"gpt2\" loads a BpeTokenizer");
        BpeTokenizer tokenizer = std::get<BpeTokenizer>(loaded);

        expect(tokenizer.vocab_size() == 258, "vocab_size is 256 base bytes + 2 merges");
        expect(tokenizer.token_bytes(256) == "ab", "merged token 256 decodes to raw bytes 'ab'");
        expect(tokenizer.token_bytes(257) == " a", "merged token 257 decodes to raw bytes ' a'");

        expect(tokenizer.encode("ab") == std::vector<std::size_t>{256}, "encode('ab') applies the learned merge");
        expect(tokenizer.encode(" a") == std::vector<std::size_t>{257}, "encode(' a') applies the learned merge");
        expect(tokenizer.decode(tokenizer.encode("ab")) == "ab", "decode(encode('ab')) round-trips");
        expect(tokenizer.decode(tokenizer.encode(" a")) == " a", "decode(encode(' a')) round-trips");
    }

    // --- SentencePiece tokenizer (tokenizer.ggml.model == "llama"): tokens + scores ---
    {
        // "hi" is reached in two merge steps: "h"+"i"->"hi", then "\xE2\x96\x81"+"hi"->"\xE2\x96\x81hi".
        std::vector<std::string> tokens = {"\xE2\x96\x81", "h", "i", "hi", "\xE2\x96\x81hi"};
        std::vector<float> scores = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};

        GgufBufferBuilder builder;
        builder.add_string_kv("tokenizer.ggml.model", "llama");
        builder.add_string_array_kv("tokenizer.ggml.tokens", tokens);
        builder.add_float32_array_kv("tokenizer.ggml.scores", scores);

        GgufReader reader(builder.build());
        mini_inference::loader::Tokenizer loaded = mini_inference::loader::build_tokenizer(reader);
        expect(std::holds_alternative<SentencePieceTokenizer>(loaded),
               "tokenizer.ggml.model == \"llama\" loads a SentencePieceTokenizer");
        const SentencePieceTokenizer &tokenizer = std::get<SentencePieceTokenizer>(loaded);

        expect(tokenizer.vocab_size() == 5, "sentencepiece vocab_size matches the tokens array");
        expect(tokenizer.decode(tokenizer.encode("hi")) == "hi", "sentencepiece decode(encode('hi')) round-trips");
    }

    // --- unsupported tokenizer model ---
    {
        GgufBufferBuilder builder;
        builder.add_string_kv("tokenizer.ggml.model", "bert");
        GgufReader reader(builder.build());

        bool threw = false;
        try
        {
            (void)mini_inference::loader::build_tokenizer(reader);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        expect(threw, "unsupported tokenizer.ggml.model throws invalid_argument");
    }

    // --- SentencePiece missing tokenizer.ggml.scores ---
    {
        GgufBufferBuilder builder;
        builder.add_string_kv("tokenizer.ggml.model", "llama");
        builder.add_string_array_kv("tokenizer.ggml.tokens", {"a", "b"});
        GgufReader reader(builder.build());

        bool threw = false;
        try
        {
            (void)mini_inference::loader::build_tokenizer(reader);
        }
        catch (const std::out_of_range &)
        {
            threw = true;
        }
        expect(threw, "missing tokenizer.ggml.scores throws out_of_range");
    }

    // --- missing tokenizer model key ---
    {
        GgufBufferBuilder builder;
        GgufReader reader(builder.build());

        bool threw = false;
        try
        {
            (void)mini_inference::loader::build_tokenizer(reader);
        }
        catch (const std::out_of_range &)
        {
            threw = true;
        }
        expect(threw, "missing tokenizer.ggml.model key throws out_of_range");
    }

    // --- malformed merge line (no separating space) ---
    {
        GgufBufferBuilder builder;
        builder.add_string_kv("tokenizer.ggml.model", "gpt2");
        builder.add_string_array_kv("tokenizer.ggml.tokens", {"a", "b"});
        builder.add_string_array_kv("tokenizer.ggml.merges", {"ab"});
        GgufReader reader(builder.build());

        bool threw = false;
        try
        {
            (void)mini_inference::loader::build_tokenizer(reader);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        expect(threw, "malformed merge entry with no space throws invalid_argument");
    }

    // --- merge referencing an unknown token ---
    {
        GgufBufferBuilder builder;
        builder.add_string_kv("tokenizer.ggml.model", "gpt2");
        builder.add_string_array_kv("tokenizer.ggml.tokens", {"a", "b"});
        builder.add_string_array_kv("tokenizer.ggml.merges", {"a c"});
        GgufReader reader(builder.build());

        bool threw = false;
        try
        {
            (void)mini_inference::loader::build_tokenizer(reader);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        expect(threw, "merge entry referencing an unknown token throws invalid_argument");
    }

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All gguf_tokenizer_loader tests passed" << std::endl;
    return 0;
}
