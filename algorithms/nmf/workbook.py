"""NMF workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_low_rank_data, relative_error  # noqa: E402


class NMF:
    """TODO: NMF by Lee and Seung multiplicative updates."""

    def __init__(self, n_components=4, seed=0, max_iter=1000, tol=1e-7):
        self.n_components = n_components
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        """TODO: reject negative entries, start W and H from a
        seeded random draw scaled by the mean of X, then repeat:
        update H by the ratio of W.T @ X to W.T @ W @ H, update W
        by the ratio of X @ H.T to W @ H @ H.T, track the relative
        reconstruction error, and stop when it barely moves. Store
        the factors in self.W_ and self.H_, the final error in
        self.reconstruction_error_, and return self."""
        raise NotImplementedError("TODO: implement fit")

    def transform(self, X):
        """TODO: check the shape of X against the fitted factors
        and return self.W_ @ self.H_."""
        raise NotImplementedError("TODO: implement transform")

    def fit_transform(self, X):
        """TODO: fit on X and return (self.W_, self.H_)."""
        raise NotImplementedError("TODO: implement fit_transform")


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
    X, W_true, H_true = make_low_rank_data(seed=0, rank=4)
    model = NMF(n_components=4, seed=0)
    results = [
        _check("fit", lambda: model.fit(X)),
        _check("transform", lambda: model.transform(X)),
        _check("fit_transform", lambda: model.fit_transform(X)),
    ]
    if all(results):
        W, H = model.fit_transform(X)
        print(f"relative error: {relative_error(X, W @ H):.4f}")
        print("All parts implemented. Now run: pytest test_nmf.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
