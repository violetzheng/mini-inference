#include "loader/gguf_loader.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <model.gguf> [prompt] [max_new_tokens]" << std::endl;
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string prompt = argc >= 3 ? argv[2] : "How are you";
    const std::size_t max_new_tokens = argc >= 4 ? std::stoul(argv[3]) : 20;

    try
    {
        std::cerr << "Loading " << model_path << "..." << std::endl;
        mini_inference::loader::GgufCheckpoint checkpoint = mini_inference::loader::load_gguf_checkpoint(model_path);

        std::vector<std::size_t> prompt_ids =
            std::visit([&](auto &tokenizer) { return tokenizer.encode(prompt); }, checkpoint.tokenizer);
        if (checkpoint.bos_token_id.has_value())
        {
            prompt_ids.insert(prompt_ids.begin(), *checkpoint.bos_token_id);
        }

        std::cerr << "Generating..." << std::endl;
        const std::vector<std::size_t> output_ids =
            checkpoint.model.generate(prompt_ids, max_new_tokens, checkpoint.eos_token_id);

        const std::string text =
            std::visit([&](auto &tokenizer) { return tokenizer.decode(output_ids); }, checkpoint.tokenizer);
        std::cout << text << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
