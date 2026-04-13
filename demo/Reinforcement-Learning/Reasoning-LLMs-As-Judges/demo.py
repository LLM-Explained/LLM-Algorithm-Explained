from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple


random.seed(0)


# ------------------------------------------------------------
# Reasoning-judge RL backbone demo
# ------------------------------------------------------------
# Simplified algorithm implemented:
#   1) a policy proposes responses
#   2) a judge scores them
#   3) the policy is updated to increase judged reward
#
# We compare:
#   - a shallow non-reasoning judge
#   - a reasoning-style judge
#
# The demo also includes an "adversarial style" response type that can
# exploit judge weaknesses, illustrating reward hacking dynamics.
# ------------------------------------------------------------


@dataclass
class Response:
    kind: str   # "good", "bad", "adversarial"
    text: str


class Policy:
    """
    Simple categorical policy over 3 response styles.
    """

    def __init__(self):
        self.probs = {
            "good": 0.34,
            "bad": 0.33,
            "adversarial": 0.33,
        }

    def sample(self) -> Response:
        r = random.random()
        cum = 0.0
        for kind, p in self.probs.items():
            cum += p
            if r <= cum:
                if kind == "good":
                    return Response(kind="good", text="helpful, honest, task-relevant answer")
                if kind == "bad":
                    return Response(kind="bad", text="low-quality off-topic answer")
                return Response(kind="adversarial", text="polished but manipulative answer with judge-pleasing cues")
        return Response(kind="bad", text="fallback")

    def update(self, rewards: List[Tuple[str, float]], lr: float = 0.1) -> None:
        """
        REINFORCE-style categorical update on expected reward.
        """
        avg = {k: 0.0 for k in self.probs}
        cnt = {k: 0 for k in self.probs}

        for kind, r in rewards:
            avg[kind] += r
            cnt[kind] += 1

        for k in avg:
            if cnt[k] > 0:
                avg[k] /= cnt[k]

        baseline = sum(r for _, r in rewards) / len(rewards)

        for k in self.probs:
            advantage = avg[k] - baseline if cnt[k] > 0 else 0.0
            self.probs[k] *= (1.0 + lr * advantage)

        # normalize
        s = sum(max(v, 1e-6) for v in self.probs.values())
        for k in self.probs:
            self.probs[k] = max(self.probs[k], 1e-6) / s


def non_reasoning_judge(response: Response) -> float:
    """
    Shallow judge:
    easily fooled by style / surface polish.
    """
    if response.kind == "good":
        return 0.75
    if response.kind == "bad":
        return 0.15
    return 0.90   # overly rewards polished adversarial output


def reasoning_judge(response: Response) -> float:
    """
    Reasoning judge:
    better at recognizing true quality, but still not perfect.
    """
    if response.kind == "good":
        return 0.92
    if response.kind == "bad":
        return 0.10
    return 0.65   # still somewhat fooled, but much less


def gold_evaluator(response: Response) -> float:
    """
    Gold-standard evaluator used only for analysis.
    """
    if response.kind == "good":
        return 1.0
    if response.kind == "bad":
        return 0.0
    return 0.20   # adversarial outputs are not truly good


def train(policy: Policy, judge_fn, steps: int = 20, batch_size: int = 64):
    history = []

    for step in range(steps):
        batch_rewards = []
        gold_scores = []
        counts = {"good": 0, "bad": 0, "adversarial": 0}

        for _ in range(batch_size):
            resp = policy.sample()
            counts[resp.kind] += 1
            jr = judge_fn(resp)
            gr = gold_evaluator(resp)

            batch_rewards.append((resp.kind, jr))
            gold_scores.append(gr)

        policy.update(batch_rewards, lr=0.15)

        history.append(
            {
                "step": step,
                "policy_probs": policy.probs.copy(),
                "avg_gold": sum(gold_scores) / len(gold_scores),
                "counts": counts,
            }
        )

    return history


def print_history(title: str, history):
    print(title)
    for item in history[::5]:
        p = item["policy_probs"]
        print(
            f"step={item['step']:02d} "
            f"probs(good={p['good']:.3f}, bad={p['bad']:.3f}, adv={p['adversarial']:.3f}) "
            f"avg_gold={item['avg_gold']:.3f} "
            f"counts={item['counts']}"
        )
    print()


def main():
    p1 = Policy()
    p2 = Policy()

    hist_non = train(p1, non_reasoning_judge, steps=25, batch_size=64)
    hist_reason = train(p2, reasoning_judge, steps=25, batch_size=64)

    print("=== Reasoning LLM-as-Judge backbone demo ===\n")
    print_history("Training with shallow non-reasoning judge", hist_non)
    print_history("Training with reasoning judge", hist_reason)

    print("Final policies:")
    print("Non-reasoning judge policy:", hist_non[-1]["policy_probs"])
    print("Reasoning judge policy    :", hist_reason[-1]["policy_probs"])
    print()

    print("Interpretation:")
    print("- The policy optimized against the shallow judge is more likely to shift toward adversarial-but-judge-pleasing outputs.")
    print("- The policy optimized against the reasoning judge is more likely to favor truly good outputs.")
    print("- This is the core algorithmic backbone of judge-driven RL in non-verifiable settings.")
    print("- It also shows why stronger judges can improve training while still leaving room for sophisticated reward hacking.")


if __name__ == "__main__":
    main()
