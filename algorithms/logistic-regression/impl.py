"""Logistic Regression
Binary classification with gradient descent on the log loss (numpy only).
Run: python algorithms/logistic-regression/impl.py
"""

import numpy as np


def _sigmoid(z):
    """Squash scores into probabilities, clipped so exp never overflows."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression:
    """Logistic regression trained by full-batch gradient descent.

    fit sets weights_ (including the bias as the first entry) and
    n_iter_. predict returns 0/1 labels, predict_proba returns the
    probability of class 1.
    """

    def __init__(self, lr=0.1, n_iter=1000, l2=0.0, seed=0):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.seed = seed

    def fit(self, X, y):
        """Fit by gradient descent on the log loss. Returns self."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        Xb = np.column_stack([np.ones(n), X])
        rng = np.random.default_rng(self.seed)
        w = rng.normal(0, 0.01, d + 1)
        for _ in range(self.n_iter):
            p = _sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) / n
            grad[1:] += self.l2 * w[1:] / n
            w -= self.lr * grad
        self.weights_ = w
        self.n_iter_ = self.n_iter
        return self

    def predict_proba(self, X):
        """Probability of class 1 for each row of X."""
        X = np.asarray(X, dtype=float)
        Xb = np.column_stack([np.ones(len(X)), X])
        return _sigmoid(Xb @ self.weights_)

    def predict(self, X):
        """0/1 labels, thresholded at probability 0.5."""
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 1.0, (100, 2))
    X1 = rng.normal(0, 1.0, (100, 2)) + np.array([3.0, 3.0])
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)
    model = LogisticRegression(lr=0.5, n_iter=1000).fit(X, y)
    acc = float(np.mean(model.predict(X) == y))
    print("weights:", np.round(model.weights_, 3))
    print("train accuracy:", round(acc, 3))
