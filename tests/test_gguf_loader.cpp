#include "loader/gguf_loader.h"
#include "gguf_test_helpers.h"
#include "loader/gpt2_byte_encoding.h"
#include "tensor/tensor.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using mini_inference::loader::GgufCheckpoint;
using mini_inference::loader::Gpt2ByteEncoding;
using mini_inference::tensor::Tensor;
using mini_inference::tests::GgufBufferBuilder;

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
    // Writes a real .gguf file and drives load_gguf_checkpoint() end-to-end.
    GgufBufferBuilder builder;

    builder.add_string_kv("general.architecture", "llama");
    builder.add_uint32_kv("llama.embedding_length", 2);
    builder.add_uint32_kv("llama.block_count", 1);
    builder.add_uint32_kv("llama.feed_forward_length", 1);
    builder.add_uint32_kv("llama.attention.head_count", 1);
    builder.add_uint32_kv("llama.attention.head_count_kv", 1);
    builder.add_float32_kv("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    builder.add_float32_kv("llama.rope.freq_base", 10000.0f);
    builder.add_uint32_kv("llama.context_length", 8);

    builder.add_tensor_f32("token_embd.weight", {2, 3}, {1.0f, 2.0f, 0.0f, 0.0f, -1.0f, -1.0f});
    builder.add_tensor_f32("blk.0.attn_norm.weight", {2}, {1.0f, 1.0f});
    builder.add_tensor_f32("blk.0.attn_q.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    builder.add_tensor_f32("blk.0.attn_k.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    builder.add_tensor_f32("blk.0.attn_v.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    builder.add_tensor_f32("blk.0.attn_output.weight", {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    builder.add_tensor_f32("blk.0.ffn_norm.weight", {2}, {1.0f, 1.0f});
    builder.add_tensor_f32("blk.0.ffn_gate.weight", {2, 1}, {1.0f, 0.0f});
    builder.add_tensor_f32("blk.0.ffn_up.weight", {2, 1}, {0.0f, 1.0f});
    builder.add_tensor_f32("blk.0.ffn_down.weight", {1, 2}, {2.0f, -1.0f});
    builder.add_tensor_f32("output_norm.weight", {2}, {1.0f, 1.0f});
    builder.add_tensor_f32("output.weight", {2, 3}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f});

    const Gpt2ByteEncoding encoding;
    std::vector<std::string> tokens;
    tokens.reserve(256);
    for (int byte = 0; byte < 256; ++byte)
    {
        tokens.push_back(encoding.encode(std::string(1, static_cast<char>(byte))));
    }
    builder.add_string_kv("tokenizer.ggml.model", "gpt2");
    builder.add_string_array_kv("tokenizer.ggml.tokens", tokens);
    builder.add_uint32_kv("tokenizer.ggml.bos_token_id", 1);
    builder.add_uint32_kv("tokenizer.ggml.eos_token_id", 2);

    const std::string path = "test_gguf_loader_tmp.gguf";
    {
        const std::vector<std::uint8_t> buffer = builder.build();
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    }

    {
        GgufCheckpoint checkpoint = mini_inference::loader::load_gguf_checkpoint(path);

        expect(checkpoint.model.vocab_size() == 3, "loaded model has the expected vocab_size");
        expect(checkpoint.tokenizer.vocab_size() == 256, "loaded tokenizer has the 256-entry base vocab");
        expect(checkpoint.bos_token_id.has_value() && *checkpoint.bos_token_id == 1,
               "bos_token_id is read from metadata");
        expect(checkpoint.eos_token_id.has_value() && *checkpoint.eos_token_id == 2,
               "eos_token_id is read from metadata");

        const Tensor output = checkpoint.model.forward({0});
        expect(output.rank() == 2 && output.shape()[0] == 1 && output.shape()[1] == 3,
               "model.forward produces logits of the expected shape");

        expect(checkpoint.tokenizer.decode(checkpoint.tokenizer.encode("hi")) == "hi",
               "tokenizer encode/decode round-trips plain text");
    }

    std::remove(path.c_str());

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed" << std::endl;
        return 1;
    }

    std::cout << "All gguf_loader tests passed" << std::endl;
    return 0;
}
