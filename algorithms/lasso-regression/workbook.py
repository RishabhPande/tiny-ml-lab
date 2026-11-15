"""Lasso Regression workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_sparse  # noqa: E402


class Lasso:
    """TODO: Lasso regression fitted by coordinate descent, with
    an unpenalized intercept."""

    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0

    def fit(self, X, y):
        """TODO: center X and y by their means. Start w at zeros
        and sweep: for each coordinate j, compute rho, the
        correlation of column j with the partial residual (the
        centered y minus the fit with coordinate j added back),
        divided by n. Set w[j] to the soft-threshold of rho at
        alpha, divided by the mean squared column value: rho
        minus alpha if rho is above alpha, rho plus alpha if
        rho is below minus alpha, and exactly 0 in between.
        Repeat sweeps until the largest change in w is under
        tol, then set coef_ to w, intercept_ to
        y_mean - x_mean @ w, n_iter_ to the sweeps used, and
        return self."""
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
    X, y, w_true = make_sparse(seed=0)
    model = Lasso(alpha=0.1)

    def _sparsity():
        if model.coef_ is None:
            raise NotImplementedError("TODO: implement fit")
        if not np.all(model.coef_[w_true == 0.0] == 0.0):
            raise AssertionError(
                "useless features are not exactly zero")

    results = [
        _check("fit", lambda: model.fit(X, y)),
        _check("predict", lambda: model.predict(X)),
        _check("sparsity: useless features exactly zero", _sparsity),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_lasso-regression.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
