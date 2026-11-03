"""Support Vector Machine: a linear classifier that finds the widest
street it can between two classes (numpy only).

The model scores a point as w.x + b and predicts +1 when the score
is positive, -1 otherwise. Training minimizes the hinge loss plus an
L2 penalty on w, by full-batch subgradient descent: each step shrinks
w toward zero (the penalty) and nudges w and b toward the points
that are inside the margin or misclassified. The result is the
maximum-margin boundary: the street between the classes is as wide
as the data allows.

Run: python algorithms/svm/impl.py
"""

import numpy as np


class SVMClassifier:
    """Linear SVM trained by full-batch subgradient descent.

    Labels must be -1 or +1 (0 is accepted and read as -1). The
    penalty lambda_ trades margin width against mistakes: a small
    lambda_ lets the model chase every point, a large one accepts
    mistakes to keep the boundary simple.
    """

    def __init__(self, lambda_=0.01, lr=0.1, n_epochs=500):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if n_epochs < 1:
            raise ValueError("n_epochs must be at least 1")
        self.lambda_ = lambda_
        self.lr = lr
        self.n_epochs = n_epochs

    def fit(self, X, y):
        """Fit the boundary. Sets coef_, intercept_, and n_steps_."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty")
        labels = np.where(y == 0, -1, y).astype(float)
        if not np.all(np.isin(labels, [-1.0, 1.0])):
            raise ValueError("y must contain only -1, 0, or +1")
        if len(np.unique(labels)) < 2:
            raise ValueError("y must contain both classes")

        w = np.zeros(X.shape[1])
        b = 0.0
        for _ in range(self.n_epochs):
            margins = labels * (X @ w + b)
            viol = margins < 1.0
            if np.any(viol):
                grad_w = self.lambda_ * w - np.mean(
                    labels[viol, None] * X[viol], axis=0
                )
                grad_b = -float(np.mean(labels[viol]))
            else:
                grad_w = self.lambda_ * w
                grad_b = 0.0
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.coef_ = w
        self.intercept_ = b
        self.n_steps_ = self.n_epochs
        return self

    def decision_function(self, X):
        """Signed distance-like score: positive means class +1."""
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict(self, X):
        """Predict -1 or +1 for each row of X."""
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def accuracy(self, X, y):
        """Fraction of correct predictions on X against labels y."""
        y = np.where(np.asarray(y) == 0, -1, np.asarray(y))
        return float(np.mean(self.predict(X) == y))


def _make_blobs(seed=42):
    """Two well separated Gaussian blobs, 100 points each."""
    rng = np.random.default_rng(seed)
    neg = rng.normal(0, 0.8, (100, 2)) + np.array([-2.0, 0.0])
    pos = rng.normal(0, 0.8, (100, 2)) + np.array([2.0, 0.0])
    X = np.vstack([neg, pos])
    y = np.array([-1] * 100 + [1] * 100)
    return X, y


if __name__ == "__main__":
    X, y = _make_blobs()
    model = SVMClassifier().fit(X, y)
    print(f"accuracy on synthetic blobs: {model.accuracy(X, y):.3f}")
    print(f"weights: {model.coef_}, intercept: {model.intercept_:.3f}")
    print(f"training epochs: {model.n_steps_}")
    scores = model.decision_function(X)
    margin_hits = np.mean(np.abs(scores) < 1.0)
    print(f"points inside the margin: {margin_hits:.3f}")
