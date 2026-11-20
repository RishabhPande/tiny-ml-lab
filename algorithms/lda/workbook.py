"""LDA workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_blobs  # noqa: E402


class LDA:
    """TODO: Linear Discriminant Analysis with nearest-centroid
    prediction in the projected space."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        """TODO: build the within-class scatter Sw and the
        between-class scatter Sb, solve eig(pinv(Sw) @ Sb), keep
        the top n_components directions (at most classes - 1) as
        unit rows in self.components_, and store the projected
        class means in self.centroids_. Return self."""
        raise NotImplementedError("TODO: implement fit")

    def transform(self, X):
        """TODO: return X projected on the discriminant
        directions, X @ self.components_.T."""
        raise NotImplementedError("TODO: implement transform")

    def predict(self, X):
        """TODO: project X, then return the class of the nearest
        centroid for each row."""
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
    X, y = make_blobs(seed=0)
    model = LDA(n_components=2)
    results = [
        _check("fit", lambda: model.fit(X, y)),
        _check("transform", lambda: model.transform(X)),
        _check("predict", lambda: model.predict(X)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_lda.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
