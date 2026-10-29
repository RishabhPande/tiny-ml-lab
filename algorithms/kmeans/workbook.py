"""K-means workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class KMeans:
    """TODO: k-means with k-means++ init and Lloyd's algorithm."""

    def __init__(self, n_clusters=3, max_iter=100, seed=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X):
        """TODO: run Lloyd's algorithm. Set centers_, labels_, inertia_."""
        raise NotImplementedError("TODO: implement fit")

    def predict(self, X):
        """TODO: assign each point to its nearest center."""
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
    rng = np.random.default_rng(0)
    X = np.vstack(
        [c + rng.normal(0, 0.7, (30, 2)) for c in [[0, 0], [5, 0], [2.5, 4]]]
    )
    km = KMeans(n_clusters=3, seed=0)
    results = [
        _check("fit", lambda: km.fit(X)),
        _check("predict", lambda: km.predict(X[:5])),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_kmeans.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
