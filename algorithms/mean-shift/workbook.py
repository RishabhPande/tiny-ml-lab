"""Mean shift workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import cluster_accuracy, make_blobs  # noqa: E402


class MeanShift:
    """TODO: mean shift clustering with a Gaussian kernel."""

    def __init__(self, bandwidth=2.0, seed=0, max_iter=100, tol=1e-3):
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self.bandwidth = bandwidth
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        """TODO: walk every point uphill by repeating the kernel
        weighted mean of all points, with weights
        exp(-distance squared / (2 * bandwidth squared)), until
        a step moves less than tol or max_iter steps run out.
        Merge endpoints closer than half the bandwidth into one
        centre, store the centres in self.cluster_centers_, the
        nearest centre index per point in self.labels_, and the
        centre count in self.n_clusters_, and return self."""
        raise NotImplementedError("TODO: implement fit")

    def fit_predict(self, X):
        """TODO: fit on X and return self.labels_."""
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
    X, y, centres = make_blobs(seed=0)
    model = MeanShift(bandwidth=2.0, seed=0)
    results = [
        _check("fit", lambda: model.fit(X)),
        _check("fit_predict", lambda: model.fit_predict(X)),
    ]
    if all(results):
        labels = model.fit_predict(X)
        print(f"clusters found: {model.n_clusters_}")
        print(f"cluster accuracy: {cluster_accuracy(y, labels):.3f}")
        print("All parts implemented. Now run: pytest test_mean_shift.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
