"""FastICA workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_mixed_signals, matched_correlation  # noqa: E402


class FastICA:
    """TODO: FastICA with whitening, a tanh contrast, and
    symmetric decorrelation at every step."""

    def __init__(self, seed=0, max_iter=500, tol=1e-6):
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        """TODO: center X into self.mean_, build the whitening
        matrix from the eigendecomposition of the covariance into
        self.whiten_, then iterate: project the whitened data on
        the current unmixing rows, apply tanh and its derivative
        mean, rebuild the rows, symmetrically decorrelate them,
        and stop when the rows barely move. Store the final rows
        in self.unmixing_ and return self."""
        raise NotImplementedError("TODO: implement fit")

    def transform(self, X):
        """TODO: center X with self.mean_, whiten it, and apply
        self.unmixing_ so each row comes back as estimated
        source values."""
        raise NotImplementedError("TODO: implement transform")

    def fit_transform(self, X):
        """TODO: fit on X and return transform(X)."""
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
    X, S, A = make_mixed_signals(seed=0)
    model = FastICA(seed=0)
    results = [
        _check("fit", lambda: model.fit(X)),
        _check("transform", lambda: model.transform(X)),
        _check("fit_transform", lambda: model.fit_transform(X)),
    ]
    if all(results):
        est = model.fit_transform(X)
        print(f"matched correlation: {matched_correlation(S, est):.3f}")
        print("All parts implemented. Now run: pytest test_ica.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
