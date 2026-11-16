"""Elastic Net workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_grouped  # noqa: E402


class ElasticNet:
    """TODO: Elastic net regression fitted by coordinate descent,
    with an unpenalized intercept."""

    def __init__(self, alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0

    def fit(self, X, y):
        """TODO: center X and y by their means. Set l1 to
        alpha * l1_ratio and l2 to alpha * (1 - l1_ratio).
        Start w at zeros and sweep: for each coordinate j,
        compute rho, the correlation of column j with the
        partial residual (the centered y minus the fit with
        coordinate j added back), divided by n. Set w[j] to
        the soft-threshold of rho at l1, divided by the mean
        squared column value plus l2: rho minus l1 if rho is
        above l1, rho plus l1 if rho is below minus l1, and
        exactly 0 in between, all over that denominator.
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
    X, y, w_true = make_grouped(seed=0)
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)

    def _grouping():
        if model.coef_ is None:
            raise NotImplementedError("TODO: implement fit")
        group = model.coef_[:3]
        if not np.max(group) - np.min(group) < 0.1:
            raise AssertionError(
                "the group effect is not shared almost evenly")

    results = [
        _check("fit", lambda: model.fit(X, y)),
        _check("predict", lambda: model.predict(X)),
        _check("grouping: near copies share the effect", _grouping),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_elastic-net.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
