"""AdaBoost workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import _fit_stump, _stump_predict, make_blobs  # noqa: E402


class AdaBoost:
    """TODO: AdaBoost over decision stumps, labels in {-1, +1}."""

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps = []
        self.alphas = np.array([])
        self.errors = []

    def fit(self, X, y):
        """TODO: start from uniform weights. Each round, fit the
        best weighted stump with _fit_stump, set its vote to
        learning_rate * 0.5 * log((1 - err) / err) with err clipped
        away from 0 and 1, stop early if err >= 0.5 (keeping the
        stump only if it is the first), then multiply the weights
        by exp(-alpha * y * pred), renormalize, and record the
        stump, vote, and error. Return self."""
        raise NotImplementedError("TODO: implement fit")

    def decision(self, X):
        """Weighted vote of the stumps for each row of X."""
        X = np.asarray(X, dtype=float)
        scores = np.zeros(X.shape[0])
        for stump, alpha in zip(self.stumps, self.alphas):
            scores = scores + alpha * _stump_predict(stump, X)
        return scores

    def predict(self, X):
        """TODO: return +1 where the vote is >= 0, else -1."""
        raise NotImplementedError("TODO: implement predict")

    def score(self, X, y):
        """Accuracy of the ensemble on (X, y)."""
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
    model = AdaBoost(n_estimators=50)
    results = [
        _check("fit", lambda: model.fit(X, y)),
        _check("predict", lambda: model.predict(X)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_adaboost.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
