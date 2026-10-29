"""Logistic regression workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def _sigmoid(z):
    """TODO: squash scores into probabilities between 0 and 1."""
    raise NotImplementedError("TODO: implement _sigmoid")


class LogisticRegression:
    """TODO: logistic regression trained by gradient descent on log loss."""

    def __init__(self, lr=0.1, n_iter=1000, l2=0.0, seed=0):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.seed = seed

    def fit(self, X, y):
        """TODO: fit by gradient descent. Set weights_ and n_iter_."""
        raise NotImplementedError("TODO: implement fit")

    def predict_proba(self, X):
        """TODO: return the probability of class 1 for each row of X."""
        raise NotImplementedError("TODO: implement predict_proba")

    def predict(self, X):
        """TODO: return 0/1 labels, thresholded at probability 0.5."""
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
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 1.0, (50, 2))
    X1 = rng.normal(0, 1.0, (50, 2)) + np.array([3.0, 3.0])
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    model = LogisticRegression(lr=0.5, n_iter=1000)
    results = [
        _check("fit", lambda: model.fit(X, y)),
        _check("predict_proba", lambda: model.predict_proba(X[:5])),
        _check("predict", lambda: model.predict(X[:5])),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_logistic-regression.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
