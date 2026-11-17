"""Hierarchical Clustering workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_blobs  # noqa: E402


class AgglomerativeClustering:
    """TODO: Agglomerative clustering with single linkage and
    an n_clusters stopping rule."""

    def __init__(self, n_clusters=2, linkage="single",
                 distance_threshold=None):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.labels_ = None
        self.n_clusters_ = 0

    def fit(self, X):
        """TODO: build the pairwise distance matrix. Start
        with one cluster per point. While more than
        n_clusters clusters remain, find the pair of clusters
        whose nearest two points (single linkage) are closest
        together, and merge them. Then assign each point the
        index of its final cluster, set labels_ and
        n_clusters_, and return self."""
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
    X, y = make_blobs(seed=0)
    model = AgglomerativeClustering(n_clusters=3, linkage="single")

    def _pure():
        if model.labels_ is None:
            raise NotImplementedError("TODO: implement fit")
        for k in range(3):
            if len(np.unique(y[model.labels_ == k])) != 1:
                raise AssertionError("a found cluster mixes blobs")

    results = [
        _check("fit", lambda: model.fit(X)),
        _check("fit_predict", lambda: model.fit_predict(X)),
        _check("purity: each cluster holds one blob", _pure),
    ]
    if all(results):
        print("All parts implemented. Now run: "
              "pytest test_hierarchical-clustering.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
