from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------
# Reasoning Cache backbone demo with a pretrained HF LLM
# ------------------------------------------------------------
# Core algorithm implemented here:
#   1) split a long problem into short-horizon chunks
#   2) ask the LLM to reason over one chunk at a time
#   3) summarize the useful progress into a compact cache
#   4) continue reasoning conditioned on the cache
#
# This is a simplified implementation of the RC-style outer loop:
# iterative reasoning + persistent compact state.
#
# Suggested default model:
#   - "Qwen/Qwen2.5-1.5B-Instruct"  (good if available locally)
#   - or a smaller instruct model you already have cached
#
# You can replace MODEL_NAME with any causal instruct model supported by
# Hugging Face Transformers.
# ------------------------------------------------------------


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class CacheState:
    partial_sum: int
    next_index: int
    steps_completed: int


def load_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 160,
    temperature: float = 0.0,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()


def make_problem() -> List[int]:
    return [17, 26, 39, 18, 24, 31, 12, 45, 28, 19, 33, 41]


def one_shot_prompt(numbers: List[int]) -> str:
    nums = ", ".join(map(str, numbers))
    return f"""You are solving a long arithmetic aggregation problem.

Problem:
Numbers: [{nums}]

Please solve it in one pass.
Return:
1. short reasoning
2. FINAL_ANSWER: <integer>
"""


def rc_reason_step_prompt(numbers: List[int], cache: CacheState, chunk_size: int = 3) -> Tuple[str, List[int], int, int]:
    start = cache.next_index
    end = min(len(numbers), start + chunk_size)
    chunk = numbers[start:end]

    prompt = f"""You are part of an iterative reasoning system.

We are summing a long list of numbers using a short-horizon reasoning step.

Current cache:
- partial_sum = {cache.partial_sum}
- next_index = {cache.next_index}
- steps_completed = {cache.steps_completed}

Current chunk:
{chunk}

Your job:
1. add the chunk
2. update the partial sum
3. return a compact structured summary

Return exactly in this format:
LOCAL_REASONING: <short text>
UPDATED_PARTIAL_SUM: <integer>
UPDATED_NEXT_INDEX: <integer>
"""
    return prompt, chunk, start, end


def extract_int(pattern: str, text: str, default: int) -> int:
    m = re.search(pattern, text)
    if m:
        return int(m.group(1))
    return default


def run_one_shot(tokenizer, model, numbers: List[int]) -> Tuple[str, int]:
    prompt = one_shot_prompt(numbers)
    text = generate_text(tokenizer, model, prompt, max_new_tokens=200)
    ans = extract_int(r"FINAL_ANSWER:\s*(-?\d+)", text, default=0)
    return text, ans


def run_reasoning_cache(tokenizer, model, numbers: List[int], chunk_size: int = 3):
    cache = CacheState(partial_sum=0, next_index=0, steps_completed=0)
    logs = []

    while cache.next_index < len(numbers):
        prompt, chunk, start, end = rc_reason_step_prompt(
            numbers, cache, chunk_size)
        out = generate_text(tokenizer, model, prompt, max_new_tokens=120)

        updated_partial = extract_int(
            r"UPDATED_PARTIAL_SUM:\s*(-?\d+)",
            out,
            default=cache.partial_sum + sum(chunk),
        )
        updated_next_index = extract_int(
            r"UPDATED_NEXT_INDEX:\s*(-?\d+)",
            out,
            default=end,
        )

        logs.append(
            {
                "cache_before": cache,
                "chunk": chunk,
                "llm_output": out,
                "cache_after": CacheState(
                    partial_sum=updated_partial,
                    next_index=updated_next_index,
                    steps_completed=cache.steps_completed + 1,
                ),
            }
        )

        cache = CacheState(
            partial_sum=updated_partial,
            next_index=updated_next_index,
            steps_completed=cache.steps_completed + 1,
        )

        # safety against malformed generations
        if cache.next_index <= start:
            cache = CacheState(
                partial_sum=cache.partial_sum,
                next_index=end,
                steps_completed=cache.steps_completed,
            )

    return logs, cache


def main():
    tokenizer, model = load_model(MODEL_NAME)

    numbers = make_problem()
    true_answer = sum(numbers)

    print("=== Reasoning Cache backbone demo with a pretrained LLM ===\n")
    print(f"Model        : {MODEL_NAME}")
    print(f"Problem      : {numbers}")
    print(f"True answer  : {true_answer}\n")

    # --------------------------------------------------------
    # One-shot baseline
    # --------------------------------------------------------
    one_shot_text, one_shot_answer = run_one_shot(tokenizer, model, numbers)

    print("----- One-shot reasoning -----")
    print(one_shot_text)
    print(f"\nOne-shot extracted answer: {one_shot_answer}\n")

    # --------------------------------------------------------
    # RC-style iterative reasoning
    # --------------------------------------------------------
    logs, final_cache = run_reasoning_cache(
        tokenizer, model, numbers, chunk_size=3)

    print("----- RC-style iterative reasoning -----")
    for i, item in enumerate(logs):
        cb = item["cache_before"]
        ca = item["cache_after"]
        print(f"\nIteration {i}")
        print(
            f"Cache before: partial_sum={cb.partial_sum}, "
            f"next_index={cb.next_index}, "
            f"steps_completed={cb.steps_completed}"
        )
        print(f"Chunk       : {item['chunk']}")
        print("LLM output  :")
        print(item["llm_output"])
        print(
            f"Cache after : partial_sum={ca.partial_sum}, "
            f"next_index={ca.next_index}, "
            f"steps_completed={ca.steps_completed}"
        )

    print("\nFinal RC cache:")
    print(final_cache)
    print(f"RC final answer from cache: {final_cache.partial_sum}\n")

    print("Interpretation:")
    print("- One-shot reasoning tries to solve the full problem in one decoding event.")
    print("- RC-style reasoning solves short-horizon chunks and carries forward only compact state.")
    print("- The cache, not the full chain, is the persistent reasoning object.")
    print("- This is the core algorithmic backbone of Reasoning Cache.")


if __name__ == "__main__":
    main()
