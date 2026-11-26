"""Factor analysis workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_factor_data, reconstruction_r2  # noqa: E402


class FactorAnalysis:
    """TODO: Factor analysis fitted by expectation-maximisation."""

    def __init__(self, n_factors=2, max_iter=500, tol=1e-7):
        if n_factors < 1:
            raise ValueError("n_factors must be at least 1")
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        """TODO: centre X into self.mean_, start the loadings
        from the top eigenvectors of the sample covariance
        scaled by the square roots of their eigenvalues, start
        the unique variances from what that guess leaves
        unexplained (floored at 0.05), then run EM rounds: the
        posterior factor covariance and mean for each point,
        loadings refit by least squares against the posteriors,
        unique variances from the leftover diagonal. Stop when
        the loadings barely move. Store self.loadings_ and
        self.noise_variance_ and return self."""
        raise NotImplementedError("TODO: implement fit")

    def transform(self, X):
        """TODO: return the posterior mean of the factors for
        each row of X, using the fitted loadings and unique
        variances."""
        raise NotImplementedError("TODO: implement transform")

    def reconstruct(self, X):
        """TODO: rebuild X from its factor scores: transform
        times the loadings transposed, plus self.mean_."""
        raise NotImplementedError("TODO: implement reconstruct")


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
    X, Z, W_true = make_factor_data(seed=0)
    model = FactorAnalysis(n_factors=2)
    results = [
        _check("fit", lambda: model.fit(X)),
        _check("transform", lambda: model.transform(X)),
        _check("reconstruct", lambda: model.reconstruct(X)),
    ]
    if all(results):
        print(f"reconstruction R2: {reconstruction_r2(X, model.reconstruct(X)):.3f}")
        print("All parts implemented. Now run: pytest test_factor_analysis.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
