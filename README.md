# mini-inference

A small, from-scratch LLM inference engine in C++, written to learn how transformer
inference actually works under the hood. Similar in spirit to llama.cpp, but optimized
for readability and simple architecture rather than production performance.

## Features so far

- **Tensor / Matrix** — minimal n-dimensional `Tensor` (shape, reshape, flat/multi-index access)
- **Linear layer** — `y = xW^T + b`.
- **RMSNorm** — root-mean-square layer normalization.
- **RoPE** — rotary position embeddings, applied per-head with a configurable position_offset
- **Softmax** — softmax over the last axis.
- **Multi-head self-attention** — scaled dot-product attention with RoPE-rotated Q/K.
- **SwiGLU** — gated feed-forward block.
- **Transformer block** — `x = x + attention(rms_norm(x))`, `x = x + swiglu(rms_norm(x))`.
- **Embedding** — token_id to hidden-vector lookup table.
- **Model** — embedding → N stacked transformer
  blocks → final RMSNorm → LM head → logits.
- **Byte-level BPE tokenizer**
- **GGUF loader** — reads a `.gguf` checkpoint (F32/F16 tensors, `llama` architecture,
  `gpt2` byte-level BPE vocab) directly into a `Model` + `BpeTokenizer`. Quantized
  tensor types and grouped-query attention checkpoints are rejected with a clear error.


## Building & testing

```sh
cmake -B build
cmake --build build
ctest --test-dir build
```

Each module (tensor, layers, tokenizer, model) has a corresponding unit test under
`tests/`.

## Project layout

```
src/
  tensor/      Tensor and Matrix primitives
  layers/      linear, rms_norm, rope, softmax, attention, swiglu, transformer_block, embedding
  model/       assembled Model (forward + generate)
  tokenizer/   byte-level BPE tokenizer
  loader/      GGUF file parsing + Model/BpeTokenizer construction
tests/         unit tests, one per module
```
