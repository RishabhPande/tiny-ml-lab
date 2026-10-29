"""Naive Bayes workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class GaussianNB:
    """TODO: Gaussian naive Bayes with independent normal features."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        """TODO: estimate priors, means, variances. Return self."""
        raise NotImplementedError("TODO: implement fit")

    def predict(self, X):
        """TODO: predict one label per row of X."""
        raise NotImplementedError("TODO: implement predict")

    def predict_proba(self, X):
        """TODO: predicted class probabilities per row of X."""
        raise NotImplementedError("TODO: implement predict_proba")


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
    X = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(0, 1, (20, 2)) + [3, 3]])
    y = np.array([0] * 20 + [1] * 20)
    nb = GaussianNB()
    results = [
        _check("fit", lambda: nb.fit(X, y)),
        _check("predict", lambda: nb.predict(X[:5])),
        _check("predict_proba", lambda: nb.predict_proba(X[:5])),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_naive-bayes.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
