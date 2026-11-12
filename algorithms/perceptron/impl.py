"""Perceptron: the original linear learner (Rosenblatt, 1958)

The perceptron keeps a weight vector w and a bias b, predicts the
sign of w @ x + b, and learns from its mistakes: whenever a point
is misclassified it nudges the weights toward that point, scaled
by its label. On data that a line can separate it is guaranteed
to stop making mistakes after finitely many updates. On data no
line can separate it never stops, because there is no line to
find. That split, perfect on separable data and hopeless on XOR,
is the whole lesson (numpy only).
Run: python algorithms/perceptron/impl.py
"""

import numpy as np


def make_blobs(seed=0, n_per_class=100, sep=2.0):
    """Two Gaussian blobs a line can separate, labels in {-1, +1}.

    Centers sit at (-sep, -sep) and (+sep, +sep) with unit spread,
    so the line x1 + x2 = 0 separates them cleanly.
    """
    rng = np.random.default_rng(seed)
    X_neg = rng.normal(-sep, 1.0, (n_per_class, 2))
    X_pos = rng.normal(sep, 1.0, (n_per_class, 2))
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([-np.ones(n_per_class), np.ones(n_per_class)])
    return X, y.astype(int)


def make_xor(seed=0, n_per_blob=50):
    """XOR blobs: same label on opposite corners, labels in {-1, +1}.

    Centers sit at (+1, +1) and (-1, -1) for the positive class
    and (+1, -1) and (-1, +1) for the negative class, with small
    spread, so no single line puts the positives on one side.
    """
    rng = np.random.default_rng(seed)
    centers_pos = [(1.0, 1.0), (-1.0, -1.0)]
    centers_neg = [(1.0, -1.0), (-1.0, 1.0)]
    X_pos = np.vstack(
        [rng.normal(c, 0.3, (n_per_blob, 2)) for c in centers_pos]
    )
    X_neg = np.vstack(
        [rng.normal(c, 0.3, (n_per_blob, 2)) for c in centers_neg]
    )
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([-np.ones(2 * n_per_blob), np.ones(2 * n_per_blob)])
    return X, y.astype(int)


def add_product_feature(X):
    """Append x1 * x2 as a third column, the feature XOR needs."""
    X = np.asarray(X, dtype=float)
    return np.column_stack([X, X[:, 0] * X[:, 1]])


class Perceptron:
    """A binary perceptron trained by the mistake-driven rule.

    Each epoch sweeps the data in a fixed-seed shuffled order and
    applies w += lr * y * x, b += lr * y to every misclassified
    point. Training stops early once an epoch makes zero mistakes,
    which on separable data is guaranteed to happen.
    """

    def __init__(self, lr=1.0, max_epochs=100, seed=0):
        self.lr = lr
        self.max_epochs = max_epochs
        self.seed = seed
        self.w = None
        self.b = 0.0
        self.mistakes_per_epoch = []
        self.converged = False

    def fit(self, X, y):
        """Train on (X, y) with labels in {-1, +1}.

        Returns the per-epoch mistake counts. Sets converged to
        True only if an epoch finishes with zero mistakes.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.mistakes_per_epoch = []
        self.converged = False
        for _ in range(self.max_epochs):
            order = rng.permutation(X.shape[0])
            mistakes = 0
            for i in order:
                if y[i] * (X[i] @ self.w + self.b) <= 0:
                    self.w = self.w + self.lr * y[i] * X[i]
                    self.b = self.b + self.lr * y[i]
                    mistakes += 1
            self.mistakes_per_epoch.append(mistakes)
            if mistakes == 0:
                self.converged = True
                break
        return self.mistakes_per_epoch

    def decision(self, X):
        """Raw scores w @ x + b for each row of X."""
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X):
        """Predicted labels in {-1, +1}: the sign of the score."""
        return np.where(self.decision(X) >= 0, 1, -1)

    def score(self, X, y):
        """Accuracy of the current weights on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


if __name__ == "__main__":
    print("Perceptron (seed 0)")

    # The works case: two separable blobs.
    X, y = make_blobs(seed=0)
    model = Perceptron(seed=0)
    history = model.fit(X, y)
    print(f"separable blobs: converged in {len(history)} epochs, "
          f"mistakes per epoch {history}, "
          f"train accuracy {model.score(X, y):.3f}")

    # The break case: XOR has no separating line.
    X, y = make_xor(seed=0)
    stuck = Perceptron(seed=0, max_epochs=50)
    history = stuck.fit(X, y)
    print(f"XOR: converged {stuck.converged} after 50 epochs, "
          f"mistakes in last epoch {history[-1]}, "
          f"train accuracy {stuck.score(X, y):.3f}")

    # Same XOR data, one extra feature: the line exists again.
    X2 = add_product_feature(X)
    fixed = Perceptron(seed=0)
    history = fixed.fit(X2, y)
    print(f"XOR with an x1*x2 feature: converged in {len(history)} "
          f"epochs, train accuracy {fixed.score(X2, y):.3f}")
