"""Linear regression workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def fit(X, y):
    """TODO: fit y ~= intercept + X @ slopes. Return [intercept, slopes...]."""
    raise NotImplementedError("TODO: implement fit")


def predict(X, coef):
    """TODO: predict with a coefficient vector from fit."""
    raise NotImplementedError("TODO: implement predict")


def mse(y_true, y_pred):
    """TODO: return the mean squared error."""
    raise NotImplementedError("TODO: implement mse")


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
    X = rng.uniform(0, 10, 20)
    y = 2.0 * X + 1.0 + rng.normal(0, 0.5, 20)
    results = [
        _check("fit", lambda: fit(X, y)),
        _check("predict", lambda: predict(X, np.array([1.0, 2.0]))),
        _check("mse", lambda: mse(y, y)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_linear-regression.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
