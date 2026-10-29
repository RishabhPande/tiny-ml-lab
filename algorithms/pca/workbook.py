"""PCA workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class PCA:
    """TODO: PCA via eigendecomposition of the covariance."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        """TODO: center X, eigendecompose, sort by variance. Return self."""
        raise NotImplementedError("TODO: implement fit")

    def transform(self, X):
        """TODO: project X onto the first n_components axes."""
        raise NotImplementedError("TODO: implement transform")

    def inverse_transform(self, Z):
        """TODO: reconstruct X from its projection Z."""
        raise NotImplementedError("TODO: implement inverse_transform")


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
    t = rng.uniform(-5, 5, 50)
    X = np.column_stack([t, 0.2 * t + rng.normal(0, 0.3, 50)])
    pca = PCA(n_components=1)
    results = [
        _check("fit", lambda: pca.fit(X)),
        _check("transform", lambda: pca.transform(X[:5])),
        _check("inverse_transform", lambda: pca.inverse_transform(np.zeros((5, 1)))),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_pca.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
