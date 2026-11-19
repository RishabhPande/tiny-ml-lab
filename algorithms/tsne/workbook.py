"""t-SNE workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_blobs  # noqa: E402


class TSNE:
    """TODO: t-SNE embedding into 2 dimensions with a fixed
    perplexity and early exaggeration."""

    def __init__(self, perplexity=20.0, n_iter=500, learning_rate=200.0,
                 exaggeration=12.0, exaggeration_iters=100, seed=0):
        self.perplexity = float(perplexity)
        self.n_iter = int(n_iter)
        self.learning_rate = float(learning_rate)
        self.exaggeration = float(exaggeration)
        self.exaggeration_iters = int(exaggeration_iters)
        self.seed = seed
        self.embedding_ = None

    def fit_transform(self, X):
        """TODO: build squared distances, binary search each
        point's Gaussian width so its neighbor entropy equals
        log2 of the perplexity, symmetrize the probabilities,
        then run gradient descent on the KL divergence with a
        Student t kernel in the map, early exaggeration for the
        first exaggeration_iters steps, adaptive gains, and
        momentum rising from 0.5 to 0.8. Recenter the map each
        step, set embedding_, and return it."""
        raise NotImplementedError("TODO: implement fit_transform")

    def fit(self, X):
        """TODO: return self after calling fit_transform."""
        raise NotImplementedError("TODO: implement fit")


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
    model = TSNE()

    def _neighborhoods_kept():
        Y = model.embedding_
        if Y is None:
            raise NotImplementedError("TODO: implement fit_transform")
        D = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(D, np.inf)
        idx = np.argsort(D, axis=1)[:, :5]
        pred = np.array([
            np.bincount(y[row], minlength=3).argmax() for row in idx])
        if float(np.mean(pred == y)) < 0.95:
            raise AssertionError("neighborhoods are not kept on the map")

    results = [
        _check("fit_transform", lambda: model.fit_transform(X)),
        _check("fit", lambda: model.fit(X)),
        _check("map keeps blob neighborhoods", _neighborhoods_kept),
    ]
    if all(results):
        print("All parts implemented. Now run: "
              "pytest test_tsne.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
