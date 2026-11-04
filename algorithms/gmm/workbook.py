"""GMM workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class GaussianMixture:
    """TODO: Gaussian mixture fitted by expectation-maximization."""

    def __init__(self, n_components=2, max_iter=100, tol=1e-4,
                 reg_covar=1e-6, seed=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.seed = seed

    def fit(self, X):
        """TODO: fit the mixture. Set weights_, means_, covariances_,
        responsibilities_, log_likelihood_, labels_, n_iter_, and
        converged_."""
        raise NotImplementedError("TODO: implement fit")

    def predict_proba(self, X):
        """TODO: responsibility of each component for each row of X."""
        raise NotImplementedError("TODO: implement predict_proba")

    def predict(self, X):
        """TODO: hard label, the most responsible component."""
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
    rng = np.random.default_rng(42)
    left = rng.normal(0, 1.0, (150, 2)) + np.array([-3.0, 0.0])
    right = rng.normal(0, 1.0, (150, 2)) + np.array([3.0, 0.0])
    X = np.vstack([left, right])
    y = np.array([0] * 150 + [1] * 150)
    holder = {}

    def _fit():
        holder["model"] = GaussianMixture().fit(X)

    def _accuracy():
        pred = holder["model"].labels_
        straight = np.mean(pred == y)
        swapped = np.mean((1 - pred) == y)
        if max(straight, swapped) < 0.95:
            raise AssertionError("accuracy below 0.95 on separated blobs")

    results = [
        _check("fit", _fit),
        _check("accuracy at least 0.95", _accuracy),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_gmm.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
