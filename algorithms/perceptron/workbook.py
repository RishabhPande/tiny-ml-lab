"""Perceptron workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_blobs  # noqa: E402


class Perceptron:
    """TODO: a binary perceptron trained by the mistake rule."""

    def __init__(self, lr=1.0, max_epochs=100, seed=0):
        self.lr = lr
        self.max_epochs = max_epochs
        self.seed = seed
        self.w = None
        self.b = 0.0
        self.mistakes_per_epoch = []
        self.converged = False

    def fit(self, X, y):
        """TODO: sweep the data in a fixed-seed shuffled order for
        up to max_epochs. On each point with y * (w @ x + b) <= 0,
        apply w += lr * y * x and b += lr * y and count a mistake.
        Record the per-epoch mistake counts, stop early and set
        converged when an epoch makes zero mistakes, and return
        the counts."""
        raise NotImplementedError("TODO: implement fit")

    def decision(self, X):
        """Raw scores w @ x + b for each row of X."""
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X):
        """TODO: return +1 where the score is >= 0, else -1."""
        raise NotImplementedError("TODO: implement predict")

    def score(self, X, y):
        """Accuracy of the current weights on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


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
    X, y = make_blobs(seed=0)
    model = Perceptron(seed=0)
    results = [
        _check("fit", lambda: model.fit(X, y)),
        _check("predict", lambda: model.predict(X)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_perceptron.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
