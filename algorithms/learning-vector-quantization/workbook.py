"""Learning vector quantization workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_blob_data  # noqa: E402


class LearningVectorQuantization:
    """TODO: LVQ1 with labelled prototypes per class."""

    def __init__(self, prototypes_per_class=2, learning_rate=0.3,
                 n_iter=3000):
        if prototypes_per_class < 1:
            raise ValueError("each class needs at least one prototype")
        self.prototypes_per_class = prototypes_per_class
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y, seed=0):
        """TODO: start the prototypes as random training points,
        the right number from each class, then repeat: decay the
        learning rate exponentially, draw one point, find its
        closest prototype, and move that prototype toward the
        point on a label match and away from it on a mismatch.
        Store self.prototypes_, self.prototype_labels_ and
        self.classes_, and return self."""
        raise NotImplementedError("TODO: implement fit")

    def predict(self, X):
        """TODO: return the label of the closest prototype for
        each row of X."""
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
    X, labels = make_blob_data(seed=0)
    model = LearningVectorQuantization(prototypes_per_class=2, n_iter=3000)
    results = [
        _check("fit", lambda: model.fit(X, labels, seed=0)),
        _check("predict", lambda: model.predict(X)),
    ]
    if all(results):
        acc = float(np.mean(model.predict(X) == labels))
        print(f"training accuracy: {acc:.3f}")
        print("All parts implemented. Now run: pytest test_learning_vector_quantization.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
