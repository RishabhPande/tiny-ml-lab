"""LoRA: Low-Rank Adaptation of a frozen linear layer (Hu et al., 2021)

Fine-tuning a big model usually means updating every weight. LoRA
freezes the pretrained weight W0 and learns only a low-rank update
dW = B @ A, where A is (r, d_in) and B is (d_out, r) with r far
smaller than d_in and d_out. A starts random, B starts at zero, so
the adapter is silent at first and the model behaves exactly like
the frozen one until training moves it. Only A and B get gradients;
W0 never changes. After training, the adapter merges back into W0
with zero extra cost at inference time (numpy only).
Run: python algorithms/lora/impl.py
"""

import numpy as np


def make_task(seed=0, d_in=16, d_out=8, n=200, rank=2, full_rank=False):
    """Synthetic adaptation task with a fixed seed.

    Returns (W0, X, Y, dW_true): a frozen random layer W0, inputs X,
    and targets Y = X @ (W0 + dW_true).T. With full_rank=False the
    true update is a rank-`rank` product, so a LoRA adapter of rank
    >= `rank` can represent it exactly. With full_rank=True the true
    update is a random full-rank matrix, which no low-rank adapter
    can represent.
    """
    rng = np.random.default_rng(seed)
    W0 = rng.normal(0.0, 1.0 / np.sqrt(d_in), (d_out, d_in))
    X = rng.normal(0.0, 1.0, (n, d_in))
    if full_rank:
        dW_true = rng.normal(0.0, 1.0 / np.sqrt(d_in), (d_out, d_in))
    else:
        U = rng.normal(0.0, 1.0, (d_out, rank))
        V = rng.normal(0.0, 1.0, (rank, d_in))
        dW_true = (U @ V) / (np.sqrt(rank) * np.sqrt(d_in))
    Y = X @ (W0 + dW_true).T
    return W0, X, Y, dW_true


class LoRALinear:
    """A frozen linear layer plus a trainable low-rank adapter.

    forward(X) = X @ W0.T + (alpha / r) * (X @ A.T) @ B.T.
    W0 is frozen; only A and B are trained, by plain gradient
    descent on mean squared error.
    """

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
        """Run the frozen layer plus the adapter on X (n, d_in)."""
        X = np.asarray(X, dtype=float)
        return X @ self.W0.T + self.scaling * ((X @ self.A.T) @ self.B.T)

    def merged_weight(self):
        """W0 plus the adapter folded in: the inference-time weight."""
        return self.W0 + self.scaling * (self.B @ self.A)

    def trainable_params(self):
        """Number of trained parameters: r * (d_in + d_out)."""
        return int(self.A.size + self.B.size)

    def full_params(self):
        """Number of parameters a full fine-tune would train."""
        return int(self.W0.size)

    def loss(self, X, Y):
        """Mean squared error of the adapted layer on (X, Y)."""
        err = self.forward(X) - np.asarray(Y, dtype=float)
        return float(np.mean(err ** 2))

    def train(self, X, Y, lr=0.1, steps=500):
        """Train only A and B by gradient descent; W0 stays frozen.

        Returns the per-step loss list (loss before each update).
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n = X.shape[0]
        losses = []
        for _ in range(steps):
            H = X @ self.A.T
            Yhat = X @ self.W0.T + self.scaling * (H @ self.B.T)
            err = Yhat - Y
            losses.append(float(np.mean(err ** 2)))
            dY = 2.0 * err / n
            dDelta = self.scaling * dY
            dB = dDelta.T @ H
            dH = dDelta @ self.B
            dA = dH.T @ X
            self.A = self.A - lr * dA
            self.B = self.B - lr * dB
        return losses


if __name__ == "__main__":
    d_in, d_out = 16, 8
    print("LoRA on a frozen 16 -> 8 layer (seed 0)")

    # The works case: the true update has rank 2, the adapter rank 4.
    W0, X, Y, dW_true = make_task(seed=0, d_in=d_in, d_out=d_out, rank=2)
    model = LoRALinear(W0, r=4, alpha=1.0, seed=0)
    frozen_loss = float(np.mean((X @ W0.T - Y) ** 2))
    print(f"frozen-only loss (rank-2 task): {frozen_loss:.4f}")
    print(f"adapter starts silent: {np.allclose(model.forward(X), X @ W0.T)}")
    model.train(X, Y, lr=0.1, steps=500)
    final = model.loss(X, Y)
    print(f"r=4 LoRA loss after training: {final:.6f} "
          f"({100.0 * final / frozen_loss:.4f}% of frozen loss)")
    print(f"trainable params: {model.trainable_params()} "
          f"vs full fine-tune: {model.full_params()}")
    merged_err = float(np.mean((model.merged_weight() - (W0 + dW_true)) ** 2))
    print(f"merged weight error vs true update: {merged_err:.2e}")

    # The break case: the true update is full rank, the adapter rank 1.
    W0, X, Y, _ = make_task(seed=0, d_in=d_in, d_out=d_out, full_rank=True)
    frozen_loss = float(np.mean((X @ W0.T - Y) ** 2))
    print(f"\nfrozen-only loss (full-rank task): {frozen_loss:.4f}")
    low = LoRALinear(W0, r=1, alpha=1.0, seed=0)
    low.train(X, Y, lr=0.1, steps=500)
    low_final = low.loss(X, Y)
    print(f"r=1 LoRA loss after training: {low_final:.4f} "
          f"({100.0 * low_final / frozen_loss:.1f}% of frozen loss remains)")
    print(f"learned update rank: {np.linalg.matrix_rank(low.merged_weight() - W0)}")
    wide = LoRALinear(W0, r=8, alpha=1.0, seed=0)
    wide.train(X, Y, lr=0.1, steps=500)
    wide_final = wide.loss(X, Y)
    print(f"r=8 LoRA loss after training: {wide_final:.4f} "
          f"({100.0 * wide_final / frozen_loss:.2f}% of frozen loss remains)")
