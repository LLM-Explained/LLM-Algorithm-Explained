# Reasoning Cache Backbone with a Pretrained LLM

This repository provides a **core backbone** for the main algorithmic idea behind **Reasoning Cache (RC)** using a pretrained causal language model from Hugging Face Transformers.

Instead of using a symbolic mockup alone, this scaffold runs a real LLM inside an **iterative reasoning loop**.

## What this code covers

This implementation includes the major algorithmic pieces:

- a **one-shot baseline**
  - the model attempts the whole problem in one decoding pass

- an **RC-style iterative loop**
  - the long problem is split into short-horizon chunks
  - the model reasons over one chunk at a time
  - useful progress is compressed into a compact cache
  - the next reasoning step is conditioned on that cache

- a **persistent compact state**
  - `partial_sum`
  - `next_index`
  - `steps_completed`

## Core algorithm

The backbone implemented here is:

1. initialize a compact cache
2. select the next short-horizon problem slice
3. ask the LLM to reason over that slice
4. extract a structured summary from the output
5. update the cache
6. continue until the full problem is solved

This captures the major RC design decision:

long-horizon reasoning is decomposed into short reasoning steps linked by compact persistent state

## Why this backbone is useful

The important part is not the arithmetic task itself.

The important part is the **outer-loop reasoning algorithm**:

- the full local reasoning trace is transient
- the cache is persistent
- progress accumulates through iterative summary-conditioned continuation

That is the key structural difference between RC-style reasoning and a one-shot decode.

## Model choice

The script assumes you can load a pretrained causal instruct model from Hugging Face.

Default:

```text
Qwen/Qwen2.5-1.5B-Instruct
```
