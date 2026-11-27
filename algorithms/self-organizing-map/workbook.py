"""Self-organizing map workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_blob_data  # noqa: E402


class SelfOrganizingMap:
    """TODO: a Kohonen self-organizing map trained by BMU updates."""

    def __init__(self, grid_rows=8, grid_cols=8, learning_rate=0.5,
                 radius=None, n_iter=4000):
        if grid_rows < 1 or grid_cols < 1:
            raise ValueError("the grid needs at least one row and one column")
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.learning_rate = learning_rate
        self.radius = radius if radius is not None else max(grid_rows, grid_cols) / 2.0
        self.n_iter = n_iter

    def fit(self, X, seed=0):
        """TODO: start the prototypes as random data points,
        then repeat: decay the learning rate and the radius
        exponentially, draw one point, find its best matching
        unit, and move every unit toward the point with a
        Gaussian weight over grid distance from that unit.
        Store the result as self.weights_ with shape
        (grid_rows, grid_cols, n_features) and return self."""
        raise NotImplementedError("TODO: implement fit")

    def bmu(self, X):
        """TODO: return the grid (row, col) of the closest
        prototype for each row of X."""
        raise NotImplementedError("TODO: implement bmu")

    def quantization_error(self, X):
        """TODO: return the mean distance from each point to
        its closest prototype."""
        raise NotImplementedError("TODO: implement quantization_error")


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
    X, labels = make_blob_data(seed=0)
    model = SelfOrganizingMap(grid_rows=8, grid_cols=8, n_iter=4000)
    results = [
        _check("fit", lambda: model.fit(X, seed=0)),
        _check("bmu", lambda: model.bmu(X)),
        _check("quantization_error", lambda: model.quantization_error(X)),
    ]
    if all(results):
        print(f"quantization error: {model.quantization_error(X):.3f}")
        print("All parts implemented. Now run: pytest test_self_organizing_map.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
