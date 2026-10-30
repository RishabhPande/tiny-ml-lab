"""Random forest workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


class RandomForestClassifier:
    """TODO: bagged trees with random feature subsets and majority vote."""

    def __init__(self, n_estimators=50, max_depth=None, min_samples_split=2,
                 max_features="sqrt", bootstrap=True, seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.seed = seed

    def fit(self, X, y):
        """TODO: fit the forest. Set classes_, trees_, feature_importances_."""
        raise NotImplementedError("TODO: implement fit")

    def predict_proba(self, X):
        """TODO: return the mean leaf fraction per class."""
        raise NotImplementedError("TODO: implement predict_proba")

    def predict(self, X):
        """TODO: return majority-vote labels."""
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
    X0 = rng.normal(0, 1.0, (50, 2))
    X1 = rng.normal(0, 1.0, (50, 2)) + np.array([3.0, 3.0])
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    model = RandomForestClassifier(n_estimators=20, seed=0)
    results = [
        _check("fit", lambda: model.fit(X, y)),
        _check("predict_proba", lambda: model.predict_proba(X[:5])),
        _check("predict", lambda: model.predict(X[:5])),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_random-forest.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
