"""Ridge Regression workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_regression  # noqa: E402


class Ridge:
    """TODO: Ridge regression with an unpenalized intercept."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        """TODO: center X and y by their means, solve
        (X^T X + alpha I) w = X^T y on the centered data with
        np.linalg.solve (fall back to np.linalg.pinv if the
        system is singular), set coef_ to the solution and
        intercept_ to y_mean - x_mean @ coef_, and return self."""
        raise NotImplementedError("TODO: implement fit")

    def predict(self, X):
        """TODO: return X @ coef_ + intercept_."""
        raise NotImplementedError("TODO: implement predict")

    def score(self, X, y):
        """R squared of the predictions on (X, y)."""
        y = np.asarray(y, dtype=float)
        pred = self.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot

    def mse(self, X, y):
        """Mean squared error of the predictions on (X, y)."""
        y = np.asarray(y, dtype=float)
        return float(np.mean((y - self.predict(X)) ** 2))


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
    X, y, _ = make_regression(seed=0)
    model = Ridge(alpha=1.0)
    results = [
        _check("fit", lambda: model.fit(X, y)),
        _check("predict", lambda: model.predict(X)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_ridge-regression.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
