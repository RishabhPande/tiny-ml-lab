"""DBSCAN workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class DBSCAN:
    """TODO: density clustering with core points, border points, noise."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        """TODO: cluster X. Set labels_, core_sample_indices_, n_clusters_."""
        raise NotImplementedError("TODO: implement fit")

    def fit_predict(self, X):
        """TODO: fit on X and return the label of each point."""
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
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.4, (50, 2))
    b = rng.normal(0, 0.4, (50, 2)) + np.array([4.0, 4.0])
    X = np.vstack([a, b])
    model = DBSCAN(eps=0.5, min_samples=5)
    results = [
        _check("fit", lambda: model.fit(X)),
        _check("fit_predict", lambda: model.fit_predict(X)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_dbscan.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
