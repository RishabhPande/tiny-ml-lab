"""LoRA workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_task  # noqa: E402


class LoRALinear:
    """TODO: a frozen linear layer plus a trainable low-rank adapter."""

    def __init__(self, W0, r=4, alpha=1.0, seed=0):
        self.W0 = np.array(W0, dtype=float)
        self.r = r
        self.alpha = alpha
        d_out, d_in = self.W0.shape
        self.d_in = d_in
        self.d_out = d_out
        rng = np.random.default_rng(seed)
        self.A = rng.normal(0.0, 0.1, (r, d_in))
        self.B = np.zeros((d_out, r))

    @property
    def scaling(self):
        """The LoRA scaling factor alpha / r."""
        return self.alpha / self.r

    def forward(self, X):
        """TODO: return X @ W0.T plus the adapter term
        (alpha / r) * (X @ A.T) @ B.T."""
        raise NotImplementedError("TODO: implement forward")

    def merged_weight(self):
        """TODO: return W0 plus the adapter folded in:
        W0 + (alpha / r) * (B @ A)."""
        raise NotImplementedError("TODO: implement merged_weight")

    def loss(self, X, Y):
        """Mean squared error of the adapted layer on (X, Y)."""
        err = self.forward(X) - np.asarray(Y, dtype=float)
        return float(np.mean(err ** 2))

    def train(self, X, Y, lr=0.1, steps=500):
        """TODO: train only A and B by gradient descent; W0 stays
        frozen. The gradients, with H = X @ A.T and err the residual
        of the adapted output, are:
          dB = (scaling * 2 * err / n).T @ H
          dA = ((scaling * 2 * err / n) @ B).T @ X
        Return the per-step loss list (loss before each update)."""
        raise NotImplementedError("TODO: implement train")


def _check(name, fn):
    try:
        fn()
    except NotImplementedError:
        print(f"[TODO] {name} is not implemented yet")
        return False
    except Exception as exc:  # noqa: BLE001 - workbook feedback, not a test
        print(f"[ERROR] {name} raised {type(exc).__name__}: {exc}")
        return False
    print(f"[OK] {name} runs")
    return True


if __name__ == "__main__":
    W0, X, Y, _ = make_task(seed=0, rank=2)
    model = LoRALinear(W0, r=4, alpha=1.0, seed=0)
    results = [
        _check("forward", lambda: model.forward(X)),
        _check("merged_weight", lambda: model.merged_weight()),
        _check("train", lambda: model.train(X, Y, lr=0.1, steps=5)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_lora.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
