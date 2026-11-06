"""Attention workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def _softmax(rows):
    """TODO: row-wise softmax, stabilized by subtracting the row max."""
    raise NotImplementedError("TODO: implement _softmax")


class SingleHeadAttention:
    """TODO: one head of scaled dot-product self-attention."""

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
        """TODO: project X into Q, K, V, score Q against K, scale by
        sqrt(d_model), softmax the rows, mix V by the weights, and
        project through W_o. Return (output, weights)."""
        raise NotImplementedError("TODO: implement forward")

    def attend_to(self, query, X):
        """TODO: project the single query with W_q and the rows of X
        with W_k and W_v, score the query against every key, softmax,
        and return the weighted value mix and the weights."""
        raise NotImplementedError("TODO: implement attend_to")


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
    rng = np.random.default_rng(0)
    attn = SingleHeadAttention(d_model=8, seed=0)
    X = rng.normal(0.0, 1.0, (6, 8))
    results = [
        _check("_softmax", lambda: _softmax(np.zeros((2, 3)))),
        _check("forward", lambda: attn.forward(X)),
        _check("attend_to", lambda: attn.attend_to(X[0], X)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_attention.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
