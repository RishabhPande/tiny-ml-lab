"""Decision tree workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class DecisionTree:
    """TODO: CART-style tree with Gini splits."""

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        """TODO: grow the tree. Return self."""
        raise NotImplementedError("TODO: implement fit")

    def predict(self, X):
        """TODO: predict one label per row of X."""
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
    X = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(0, 1, (20, 2)) + [4, 0]])
    y = np.array([0] * 20 + [1] * 20)
    tree = DecisionTree(max_depth=4)
    results = [
        _check("fit", lambda: tree.fit(X, y)),
        _check("predict", lambda: tree.predict(X[:5])),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_decision-tree.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
