"""Scaled dot-product self-attention (single head)
The core of the Transformer: every token scores every other token
with a query-key dot product, the scores become weights, and each
token's output is a weighted mix of the values (numpy only).
Run: python algorithms/attention/impl.py
"""

import numpy as np


def _softmax(rows):
    """Row-wise softmax, stabilized by subtracting the row maximum."""
    shifted = rows - rows.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class SingleHeadAttention:
    """One head of scaled dot-product self-attention.

    Each input row is projected into a query, a key, and a value.
    The attention weight from token i to token j is the softmax of
    q_i . k_j / sqrt(d_model), and token i's output is the weighted
    sum of all values, passed through an output projection.
    """

    def __init__(self, d_model=8, seed=0):
        self.d_model = d_model
        self.seed = seed
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.W_q = rng.normal(0.0, scale, (d_model, d_model))
        self.W_k = rng.normal(0.0, scale, (d_model, d_model))
        self.W_v = rng.normal(0.0, scale, (d_model, d_model))
        self.W_o = rng.normal(0.0, scale, (d_model, d_model))

    def forward(self, X):
        """Run attention on X (n_tokens, d_model).

        Returns (output, weights): output has the same shape as X,
        and weights[i, j] is how much token i attends to token j.
        Each row of weights sums to 1.
        """
        X = np.asarray(X, dtype=float)
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        scores = (Q @ K.T) / np.sqrt(self.d_model)
        weights = _softmax(scores)
        out = (weights @ V) @ self.W_o
        return out, weights

    def attend_to(self, query, X):
        """Attention of one outside query over the rows of X.

        Returns (output, weights): output is a single d_model
        vector, and weights[j] is how much the query attends to
        token j. This is the retrieval view of the same mechanism:
        the query finds the key it matches best.
        """
        query = np.asarray(query, dtype=float)
        X = np.asarray(X, dtype=float)
        q = query @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        scores = (K @ q) / np.sqrt(self.d_model)
        weights = _softmax(scores.reshape(1, -1))[0]
        out = (weights @ V) @ self.W_o
        return out, weights


def _demo_tokens(rng, n_tokens=6, d_model=8):
    """Six tokens plus a query that matches only token 4."""
    query = rng.normal(0.0, 1.0, d_model)
    X = rng.normal(0.0, 1.0, (n_tokens, d_model))
    X[4] = query + rng.normal(0.0, 0.05, d_model)
    return query, X


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d_model = 16
    attn = SingleHeadAttention(d_model=d_model, seed=0)
    # Identity projections, so Q = K = V = X and the demo shows the
    # mechanism itself, not learned weights.
    attn.W_q = np.eye(d_model)
    attn.W_k = np.eye(d_model)
    attn.W_v = np.eye(d_model)
    attn.W_o = np.eye(d_model)

    query, X = _demo_tokens(rng, n_tokens=6, d_model=d_model)
    out, weights = attn.attend_to(query, X)
    print("query attention weights over the 6 tokens (match is token 4):")
    print(np.round(weights, 3))
    print("query attends most to token:", int(np.argmax(weights)),
          "with weight", round(float(weights.max()), 3))
    print("query output is closest to token",
          int(np.argmin(np.linalg.norm(out - X, axis=1))),
          "of the inputs (the token it attended to)")

    _, self_weights = attn.forward(X)
    print("self-attention row sums (each row must sum to 1):",
          np.round(self_weights.sum(axis=1), 6))

    # The break case: every token identical, so every key matches
    # every query equally and attention goes uniform.
    X_flat = np.tile(X[0], (6, 1))
    _, flat_weights = attn.forward(X_flat)
    print("identical tokens: attention weights from token 0:",
          np.round(flat_weights[0], 3))
