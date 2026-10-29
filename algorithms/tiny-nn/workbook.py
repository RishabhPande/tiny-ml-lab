"""Tiny NN workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class MLP:
    """TODO: one-hidden-layer MLP trained by full-batch gradient descent."""

    def __init__(self, hidden=8, lr=0.5, epochs=5000, seed=0):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        """TODO: train on X with binary labels y. Return self."""
        raise NotImplementedError("TODO: implement fit")

    def predict_proba(self, X):
        """TODO: predicted probability of class 1 per row of X."""
        raise NotImplementedError("TODO: implement predict_proba")

    def predict(self, X):
        """TODO: predict 0/1 labels per row of X."""
        raise NotImplementedError("TODO: implement predict")


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
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([0, 1, 1, 0])
    mlp = MLP(hidden=8, lr=0.5, epochs=500, seed=0)
    results = [
        _check("fit", lambda: mlp.fit(X_xor, y_xor)),
        _check("predict_proba", lambda: mlp.predict_proba(X_xor)),
        _check("predict", lambda: mlp.predict(X_xor)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_tiny-nn.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
