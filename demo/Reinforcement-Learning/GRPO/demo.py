from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List


random.seed(0)


# ------------------------------------------------------------
# GRPO backbone demo
# ------------------------------------------------------------
# Major algorithmic pieces implemented:
#   1) generate a group of completions for the same prompt
#   2) score each completion with a reward function
#   3) compute group-relative normalized advantages
#   4) update the policy using group-relative signal
#
# This is a simplified implementation of the core GRPO mechanism itself.
# ------------------------------------------------------------


@dataclass
class Completion:
    kind: str
    text: str


class Policy:
    """
    Simple categorical policy over 3 completion styles.
    """

    def __init__(self):
        self.probs = {
            "strong_reasoning": 0.34,
            "medium_reasoning": 0.33,
            "weak_reasoning": 0.33,
        }

    def sample_group(self, group_size: int) -> List[Completion]:
        out = []
        for _ in range(group_size):
            r = random.random()
            cum = 0.0
            for kind, p in self.probs.items():
                cum += p
                if r <= cum:
                    if kind == "strong_reasoning":
                        out.append(Completion(
                            kind, "careful multi-step solution"))
                    elif kind == "medium_reasoning":
                        out.append(Completion(
                            kind, "partial but useful reasoning"))
                    else:
                        out.append(Completion(
                            kind, "short low-quality answer"))
                    break
        return out

    def update_from_group(self, kinds: List[str], advantages: List[float], lr: float = 0.15) -> None:
        avg_adv = {k: 0.0 for k in self.probs}
        cnt = {k: 0 for k in self.probs}

        for k, a in zip(kinds, advantages):
            avg_adv[k] += a
            cnt[k] += 1

        for k in avg_adv:
            if cnt[k] > 0:
                avg_adv[k] /= cnt[k]

        for k in self.probs:
            self.probs[k] *= (1.0 + lr * avg_adv[k])

        s = sum(max(v, 1e-6) for v in self.probs.values())
        for k in self.probs:
            self.probs[k] = max(self.probs[k], 1e-6) / s


def reward_fn(prompt: str, completion: Completion) -> float:
    """
    Simplified reward:
    strong > medium > weak
    """
    if completion.kind == "strong_reasoning":
        return 1.0 + random.uniform(-0.05, 0.05)
    if completion.kind == "medium_reasoning":
        return 0.6 + random.uniform(-0.05, 0.05)
    return 0.2 + random.uniform(-0.05, 0.05)


def group_relative_advantages(rewards: List[float]) -> List[float]:
    mu = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var) + 1e-6
    return [(r - mu) / std for r in rewards]


def main():
    prompt = "Solve the math reasoning problem carefully."
    policy = Policy()
    group_size = 6

    print("=== GRPO backbone demo ===\n")
    print(f"Initial policy probs: {policy.probs}\n")

    for step in range(10):
        completions = policy.sample_group(group_size)
        rewards = [reward_fn(prompt, c) for c in completions]
        advantages = group_relative_advantages(rewards)

        print(f"Step {step}")
        for i, (c, r, a) in enumerate(zip(completions, rewards, advantages)):
            print(
                f"  sample {i}: kind={c.kind:16s} "
                f"reward={r:.3f} "
                f"adv={a:+.3f}"
            )

        policy.update_from_group(
            kinds=[c.kind for c in completions],
            advantages=advantages,
            lr=0.12,
        )

        print(f"  updated policy probs: {policy.probs}\n")

    print("Interpretation:")
    print("- Each prompt generates a group of completions.")
    print("- Rewards are normalized within the group, not against a learned critic.")
    print("- Stronger-than-group-average completions get positive advantage.")
    print("- Weaker-than-group-average completions get negative advantage.")
    print("- This is the core GRPO mechanism: critic-free group-relative advantage estimation.")


if __name__ == "__main__":
    main()
