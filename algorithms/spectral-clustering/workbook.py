"""Spectral Clustering workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_rings  # noqa: E402


class SpectralClustering:
    """TODO: spectral clustering with a Gaussian kernel graph
    and an n_clusters stopping rule."""

    def __init__(self, n_clusters=2, sigma=0.5, seed=0):
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.seed = seed
        self.labels_ = None
        self.embedding_ = None

    def fit(self, X):
        """TODO: build the Gaussian affinity matrix with width
        sigma and a zero diagonal. Form the normalized
        Laplacian I minus D^-1/2 W D^-1/2, guarding zero
        degrees. Take the n_clusters smallest eigenvectors
        with numpy.linalg.eigh, rescale each row to unit
        length, and cluster the rows with a small k-means of
        your own (seeded restarts, keep the best inertia). Set
        labels_ and embedding_, and return self."""
        raise NotImplementedError("TODO: implement fit")

    def fit_predict(self, X):
        """TODO: return self.fit(X).labels_."""
        raise NotImplementedError("TODO: implement fit_predict")


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
    X, y = make_rings(seed=0)
    model = SpectralClustering(n_clusters=2)

    def _rings_split():
        if model.labels_ is None:
            raise NotImplementedError("TODO: implement fit")
        plain = np.mean(y == model.labels_)
        flipped = np.mean(y == (1 - model.labels_))
        if max(plain, flipped) != 1.0:
            raise AssertionError("the rings are not fully split")

    results = [
        _check("fit", lambda: model.fit(X)),
        _check("fit_predict", lambda: model.fit_predict(X)),
        _check("rings: each ring gets its own label", _rings_split),
    ]
    if all(results):
        print("All parts implemented. Now run: "
              "pytest test_spectral-clustering.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
