from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


np.random.seed(0)
random.seed(0)


# ------------------------------------------------------------
# Simplified continual-learning algorithm backbone
# ------------------------------------------------------------
# This demo implements a compact continual-learning scaffold with:
#   1) sequential task stream
#   2) replay buffer
#   3) regularized updates against old parameters
#
# It is intentionally small, but it includes the major algorithmic
# ingredients that define many continual-learning methods in practice.
# ------------------------------------------------------------


@dataclass
class TaskData:
    x: np.ndarray
    y: np.ndarray
    name: str


class TinyMLP:
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        self.W1 = 0.1 * np.random.randn(in_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = 0.1 * np.random.randn(hidden_dim, out_dim)
        self.b2 = np.zeros(out_dim)

    def copy(self) -> "TinyMLP":
        m = TinyMLP(self.W1.shape[0], self.W1.shape[1], self.W2.shape[1])
        m.W1 = self.W1.copy()
        m.b1 = self.b1.copy()
        m.W2 = self.W2.copy()
        m.b2 = self.b2.copy()
        return m

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h_pre = x @ self.W1 + self.b1
        h = np.tanh(h_pre)
        logits = h @ self.W2 + self.b2
        return h, logits

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, logits = self.forward(x)
        return np.argmax(logits, axis=1)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy_and_grads(model: TinyMLP, x: np.ndarray, y: np.ndarray):
    h, logits = model.forward(x)
    probs = softmax(logits)

    n = x.shape[0]
    loss = -np.log(np.maximum(probs[np.arange(n), y], 1e-9)).mean()

    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n

    gW2 = h.T @ dlogits
    gb2 = np.sum(dlogits, axis=0)

    dh = dlogits @ model.W2.T
    dh_pre = dh * (1.0 - h ** 2)

    gW1 = x.T @ dh_pre
    gb1 = np.sum(dh_pre, axis=0)

    return loss, (gW1, gb1, gW2, gb2)


def l2_reg_to_old(model: TinyMLP, old_model: TinyMLP):
    loss = (
        np.mean((model.W1 - old_model.W1) ** 2)
        + np.mean((model.b1 - old_model.b1) ** 2)
        + np.mean((model.W2 - old_model.W2) ** 2)
        + np.mean((model.b2 - old_model.b2) ** 2)
    )
    grads = (
        2.0 * (model.W1 - old_model.W1) / model.W1.size,
        2.0 * (model.b1 - old_model.b1) / model.b1.size,
        2.0 * (model.W2 - old_model.W2) / model.W2.size,
        2.0 * (model.b2 - old_model.b2) / model.b2.size,
    )
    return loss, grads


class ReplayBuffer:
    def __init__(self, max_items: int = 256):
        self.max_items = max_items
        self.x: List[np.ndarray] = []
        self.y: List[int] = []

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        for xi, yi in zip(x, y):
            self.x.append(xi.copy())
            self.y.append(int(yi))
        if len(self.x) > self.max_items:
            extra = len(self.x) - self.max_items
            self.x = self.x[extra:]
            self.y = self.y[extra:]

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray] | None:
        if len(self.x) == 0:
            return None
        idx = np.random.choice(len(self.x), size=min(
            n, len(self.x)), replace=False)
        xs = np.stack([self.x[i] for i in idx], axis=0)
        ys = np.array([self.y[i] for i in idx], dtype=np.int64)
        return xs, ys


def train_step(
    model: TinyMLP,
    batch_x: np.ndarray,
    batch_y: np.ndarray,
    lr: float,
    old_model: TinyMLP | None = None,
    reg_lambda: float = 0.0,
):
    task_loss, task_grads = cross_entropy_and_grads(model, batch_x, batch_y)

    gW1, gb1, gW2, gb2 = task_grads
    total_loss = task_loss

    if old_model is not None and reg_lambda > 0.0:
        reg_loss, reg_grads = l2_reg_to_old(model, old_model)
        rgW1, rgb1, rgW2, rgb2 = reg_grads

        gW1 += reg_lambda * rgW1
        gb1 += reg_lambda * rgb1
        gW2 += reg_lambda * rgW2
        gb2 += reg_lambda * rgb2
        total_loss += reg_lambda * reg_loss

    model.W1 -= lr * gW1
    model.b1 -= lr * gb1
    model.W2 -= lr * gW2
    model.b2 -= lr * gb2

    return total_loss


def accuracy(model: TinyMLP, x: np.ndarray, y: np.ndarray) -> float:
    pred = model.predict(x)
    return float(np.mean(pred == y))


def make_task(angle_deg: float, n: int = 256, noise: float = 0.15, name: str = "") -> TaskData:
    angle = np.deg2rad(angle_deg)
    w = np.array([np.cos(angle), np.sin(angle)])

    x = np.random.randn(n, 2)
    logits = x @ w + noise * np.random.randn(n)
    y = (logits > 0).astype(np.int64)

    return TaskData(x=x, y=y, name=name or f"task_{angle_deg}")


def continual_learning_run(
    tasks: List[TaskData],
    use_replay: bool,
    use_regularization: bool,
) -> None:
    model = TinyMLP(in_dim=2, hidden_dim=32, out_dim=2)
    replay = ReplayBuffer(max_items=256)
    lr = 0.1

    print("=" * 72)
    print(
        f"Setting: replay={use_replay}, "
        f"regularization={use_regularization}"
    )
    print("=" * 72)

    for task_id, task in enumerate(tasks):
        old_model = model.copy() if use_regularization else None

        for step in range(100):
            batch_x = task.x
            batch_y = task.y

            if use_replay:
                replay_batch = replay.sample(64)
                if replay_batch is not None:
                    rx, ry = replay_batch
                    batch_x = np.concatenate([batch_x, rx], axis=0)
                    batch_y = np.concatenate([batch_y, ry], axis=0)

            loss = train_step(
                model=model,
                batch_x=batch_x,
                batch_y=batch_y,
                lr=lr,
                old_model=old_model,
                reg_lambda=1.0 if use_regularization else 0.0,
            )

        replay.add(task.x[:64], task.y[:64])

        print(f"\nAfter learning {task.name}:")
        for prev_task in tasks[: task_id + 1]:
            acc = accuracy(model, prev_task.x, prev_task.y)
            print(f"  acc on {prev_task.name:8s}: {acc:.3f}")


def main() -> None:
    tasks = [
        make_task(0, name="task_A"),
        make_task(35, name="task_B"),
        make_task(70, name="task_C"),
    ]

    print("=== Continual Learning Demo ===\n")
    print("We compare three settings across a stream of sequential tasks:")
    print("1) plain sequential fine-tuning")
    print("2) continual learning with replay")
    print("3) continual learning with replay + regularization\n")

    continual_learning_run(tasks, use_replay=False, use_regularization=False)
    continual_learning_run(tasks, use_replay=True, use_regularization=False)
    continual_learning_run(tasks, use_replay=True, use_regularization=True)

    print("\nInterpretation:")
    print("- Plain sequential updates tend to forget older tasks.")
    print("- Replay preserves access to older data distributions.")
    print("- Regularization discourages destructive drift from older parameters.")
    print("- Together they form a compact continual-learning backbone.")


if __name__ == "__main__":
    main()
