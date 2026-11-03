"""SVM workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class SVMClassifier:
    """TODO: linear SVM trained by full-batch subgradient descent."""

    def __init__(self, lambda_=0.01, lr=0.1, n_epochs=500):
        self.lambda_ = lambda_
        self.lr = lr
        self.n_epochs = n_epochs

    def fit(self, X, y):
        """TODO: fit the boundary. Set coef_, intercept_, n_steps_."""
        raise NotImplementedError("TODO: implement fit")

    def decision_function(self, X):
        """TODO: return w.x + b for each row of X."""
        raise NotImplementedError("TODO: implement decision_function")

    def predict(self, X):
        """TODO: predict -1 or +1 for each row of X."""
        raise NotImplementedError("TODO: implement predict")

    def accuracy(self, X, y):
        """TODO: fraction of correct predictions."""
        raise NotImplementedError("TODO: implement accuracy")


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
    rng = np.random.default_rng(42)
    neg = rng.normal(0, 0.8, (100, 2)) + np.array([-2.0, 0.0])
    pos = rng.normal(0, 0.8, (100, 2)) + np.array([2.0, 0.0])
    X = np.vstack([neg, pos])
    y = np.array([-1] * 100 + [1] * 100)
    holder = {}

    def _fit():
        holder["model"] = SVMClassifier().fit(X, y)

    def _accuracy():
        if holder["model"].accuracy(X, y) < 0.97:
            raise AssertionError("accuracy below 0.97 on separated blobs")

    results = [
        _check("fit", _fit),
        _check("accuracy at least 0.97", _accuracy),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_svm.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
