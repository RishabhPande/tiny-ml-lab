"""QDA workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_same_mean  # noqa: E402


class QDA:
    """TODO: Quadratic Discriminant Analysis with one Gaussian
    per class, each with its own covariance."""

    def __init__(self, reg=1e-6):
        self.reg = reg

    def fit(self, X, y):
        """TODO: for each class store its mean in self.means_,
        its covariance (with a small reg ridge on the diagonal)
        in self.covs_, its prior in self.priors_, and the log
        determinant of its covariance in self.logdets_.
        Return self."""
        raise NotImplementedError("TODO: implement fit")

    def log_posterior(self, X):
        """TODO: return the log posterior of each row of X under
        every class: -0.5 * logdet - 0.5 * Mahalanobis distance
        + log prior, as an array of shape (n_rows, n_classes)."""
        raise NotImplementedError("TODO: implement log_posterior")

    def predict(self, X):
        """TODO: return the class with the highest log posterior
        for each row of X."""
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
    X, y = make_same_mean(seed=0)
    model = QDA()
    results = [
        _check("fit", lambda: model.fit(X, y)),
        _check("log_posterior", lambda: model.log_posterior(X)),
        _check("predict", lambda: model.predict(X)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_qda.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
