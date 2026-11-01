"""Gradient boosting workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class GradientBoostingRegressor:
    """TODO: stagewise regression from decision stumps."""

    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """TODO: start at the mean of y, then fit one stump per round
        to the residuals and add it, shrunk by the learning rate.
        Set init_, stumps_, train_mse_."""
        raise NotImplementedError("TODO: implement fit")

    def predict(self, X):
        """TODO: mean of y plus every shrunken stump."""
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
    X = rng.uniform(0.0, 3.0, (100, 1))
    y = np.sin(2.0 * X[:, 0]) + rng.normal(0, 0.1, 100)
    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1)
    results = [
        _check("fit", lambda: model.fit(X, y)),
        _check("predict", lambda: model.predict(X)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_gradient_boosting.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
