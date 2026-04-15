from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)


# ------------------------------------------------------------
# Muon backbone demo
# ------------------------------------------------------------
# Major algorithmic pieces implemented:
#   1) momentum update
#   2) matrix orthogonalization with Newton–Schulz iteration
#   3) Muon update for 2D hidden-layer weights
#   4) standard SGD-style updates for biases/output head
#
# This is a simplified implementation of the Muon algorithm itself.
# ------------------------------------------------------------


def zeropower_via_newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Approximate Orth(G) using the Newton–Schulz-style polynomial iteration
    used in Muon implementations.

    Input:
      G: [out_dim, in_dim] matrix
    Output:
      orthogonalized update matrix with similar singular vectors and flattened spectrum
    """
    assert G.ndim == 2

    X = G / (G.norm() + eps)

    # coefficients used in Muon-style NS5 iteration
    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * ((A @ A) @ X)

    return X


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 32, out_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


def make_xor_data(n: int = 1024):
    x = torch.rand(n, 2) * 2 - 1
    y = ((x[:, 0] * x[:, 1]) < 0).long()
    return x, y


class MuonOptimizer:
    """
    Simplified Muon optimizer:

    - apply Muon only to 2D hidden-layer weight matrices
    - use plain SGD-style updates for biases / output layer if desired
    """

    def __init__(self, hidden_weight: torch.nn.Parameter, lr: float = 0.05, momentum: float = 0.95, ns_steps: int = 5):
        self.weight = hidden_weight
        self.lr = lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.m = torch.zeros_like(hidden_weight.data)

    @torch.no_grad()
    def step(self):
        g = self.weight.grad
        if g is None:
            return

        self.m.mul_(self.momentum).add_(g)
        update = zeropower_via_newton_schulz5(self.m, steps=self.ns_steps)

        # practical scale normalization similar in spirit to Muon implementations
        update = update * max(1.0, (update.size(-2) / update.size(-1)) ** 0.5)

        self.weight.data.add_(-self.lr * update)


class SimpleSGD:
    def __init__(self, params, lr: float = 0.05):
        self.params = list(params)
        self.lr = lr

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data.add_(-self.lr * p.grad)


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
        return (pred == y).float().mean().item()


def main():
    x_train, y_train = make_xor_data(2048)
    x_test, y_test = make_xor_data(512)

    model = TinyMLP()

    # Muon on hidden-layer matrix only
    muon = MuonOptimizer(model.fc1.weight, lr=0.08, momentum=0.95, ns_steps=5)

    # plain SGD on bias and output layer for simplicity
    sgd = SimpleSGD(
        params=[model.fc1.bias, model.fc2.weight, model.fc2.bias],
        lr=0.05,
    )

    print("=== Muon backbone demo ===\n")

    for step in range(200):
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)

        model.zero_grad(set_to_none=True)
        loss.backward()

        # inspect the matrix-update geometry occasionally
        if step % 50 == 0:
            with torch.no_grad():
                g = model.fc1.weight.grad
                U, S, Vh = torch.linalg.svd(g, full_matrices=False)
                print(
                    f"step={step:03d} raw grad singular values (first 5): {S[:5].tolist()}")

                m_preview = muon.m * muon.momentum + g
                ortho = zeropower_via_newton_schulz5(m_preview, steps=5)
                U2, S2, Vh2 = torch.linalg.svd(ortho, full_matrices=False)
                print(
                    f"step={step:03d} orth update singular values (first 5): {S2[:5].tolist()}")

        muon.step()
        sgd.step()

        if step % 50 == 0 or step == 199:
            acc = accuracy(model, x_test, y_test)
            print(
                f"step={step:03d} loss={loss.item():.4f} test_acc={acc:.4f}\n")

    print("Interpretation:")
    print("- The hidden-layer weight is updated with momentum + matrix orthogonalization.")
    print("- The singular values of the orthogonalized update are much flatter than the raw gradient.")
    print("- This is the core algorithmic idea behind Muon.")


if __name__ == "__main__":
    main()
