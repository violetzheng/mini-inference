# mini-inference

A small LLM inference engine I'm building from scratch in C++, mostly to actually
understand how transformer inference works under the hood. Kind of like llama.cpp but
way simpler and not trying to be fast - readability over performance.

## What's implemented

- Tensor / Matrix - basic n-dimensional tensor, shape/reshape/indexing
- Linear layer (`y = xW^T + b`)
- RMSNorm
- RoPE, applied per-head with a configurable position offset
- Softmax
- Multi-head self-attention with RoPE-rotated Q/K
- Gated feed-forward block (SwiGLU or GeGLU gate)
- Transformer block: `x = x + attention(rms_norm(x))`, then `x = x + swiglu(rms_norm(x))`
- Embedding lookup table
- Model: embedding -> N transformer blocks -> final RMSNorm -> LM head -> logits
- Byte-level BPE and SentencePiece tokenizers
- GGUF loader - reads a `.gguf` file straight into a `Model` + tokenizer (see below for
  what it actually supports)
- A tiny CLI (`mini_inference`) that loads a checkpoint and generates from a prompt

## Building & testing

```sh
cmake -B build
cmake --build build
ctest --test-dir build
```

Every module has its own test file under `tests/`.

## What models actually work

The GGUF loader is pretty picky. For a checkpoint to load it needs:

- `general.architecture` to be `llama`, `qwen2`, or `gemma`. Other architectures (phi3, ...)
  don't work right now. Gemma support is scoped to variants where `head_dim == embedding_length / head_count` (e.g. gemma-2b) - larger ones like gemma-7b use a head_dim the loader can't derive yet and are rejected.
- Multi-head or grouped-query attention supported.
- A gpt2-style byte-BPE tokenizer or llama's SentencePiece one.
- Tensors in F32, F16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, Q2_K or Q3_K. This includes the
  embedding table itself being quantized, not just the layer weights. Still missing:
  Q4_1, Q5_0, Q5_1, Q8_1, Q8_K.

I've tested with TinyLlama-1.1B-Chat-v1.0, both the Q6_K and Q4_K_S GGUF
files. `tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf` and `tinyllama-1.1b-chat-v1.0.Q6_K.gguf`.

## Running it

```sh
./build/mini_inference <model.gguf> [prompt] [max_new_tokens]
```

`prompt` defaults to `"How are you"`, `max_new_tokens` defaults to `20`.

### Example

```
$ ./build/mini_inference gguf_files/tinyllama-1.1b-chat-v1.0.Q6_K.gguf "Once upon a time" 20
Loading gguf_files/tinyllama-1.1b-chat-v1.0.Q6_K.gguf...
Generating...
<s> Once upon a time, there was a young woman named Lily. She lived in a small town, where everyone knew
```

```
$ ./build/mini_inference gguf_files/tinyllama-1.1b-chat-v1.0.Q6_K.gguf
Loading gguf_files/tinyllama-1.1b-chat-v1.0.Q6_K.gguf...
Generating...
<s> How are you able to provide such a wide range of products and services to your customers?</s>
```

## Layout

```
src/
  main.cpp     CLI entry point
  tensor/      Tensor/Matrix, SIMD + parallel_for helpers, quantization block decoding
  layers/      linear, rms_norm, rope, softmax, attention, swiglu, transformer_block, embedding
  model/       the assembled Model (forward + generate)
  tokenizer/   BPE + SentencePiece
  loader/      GGUF parsing, builds a Model/tokenizer out of it
tests/         one test file per module
```
