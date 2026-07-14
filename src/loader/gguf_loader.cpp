#include "loader/gguf_loader.h"
#include "loader/gguf_model_loader.h"
#include "loader/gguf_reader.h"
#include "loader/gguf_tokenizer_loader.h"

namespace mini_inference::loader
{

    GgufCheckpoint load_gguf_checkpoint(const std::string &path)
    {
        GgufReader reader(path);

        GgufCheckpoint checkpoint{
            build_model(reader),
            build_tokenizer(reader),
            std::nullopt,
            std::nullopt,
        };

        if (reader.has_metadata("tokenizer.ggml.bos_token_id"))
        {
            checkpoint.bos_token_id = reader.metadata_uint32("tokenizer.ggml.bos_token_id", 0);
        }
        if (reader.has_metadata("tokenizer.ggml.eos_token_id"))
        {
            checkpoint.eos_token_id = reader.metadata_uint32("tokenizer.ggml.eos_token_id", 0);
        }

        return checkpoint;
    }

} // namespace mini_inference::loader
