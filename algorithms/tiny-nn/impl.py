"""A tiny numpy MLP: tanh hidden layer, sigmoid output (numpy only)."""

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class MLP:
    """One-hidden-layer MLP trained by full-batch gradient descent."""

    def __init__(self, hidden=8, lr=0.5, epochs=5000, seed=0):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        """Train on X with binary labels y. Returns self."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        m, n_in = X.shape
        rng = np.random.default_rng(self.seed)
        self.W1_ = rng.normal(0, np.sqrt(1.0 / n_in), (n_in, self.hidden))
        self.b1_ = np.zeros((1, self.hidden))
        self.W2_ = rng.normal(0, np.sqrt(1.0 / self.hidden), (self.hidden, 1))
        self.b2_ = np.zeros((1, 1))
        for _ in range(self.epochs):
            Z1 = X @ self.W1_ + self.b1_
            A1 = np.tanh(Z1)
            Z2 = A1 @ self.W2_ + self.b2_
            A2 = _sigmoid(Z2)
            dZ2 = (A2 - y) / m
            dW2 = A1.T @ dZ2
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            dA1 = dZ2 @ self.W2_.T
            dZ1 = dA1 * (1.0 - A1**2)
            dW1 = X.T @ dZ1
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            self.W2_ -= self.lr * dW2
            self.b2_ -= self.lr * db2
            self.W1_ -= self.lr * dW1
            self.b1_ -= self.lr * db1
        return self

    def predict_proba(self, X):
        """Predicted probability of class 1 per row of X."""
        X = np.asarray(X, dtype=float)
        A1 = np.tanh(X @ self.W1_ + self.b1_)
        return _sigmoid(A1 @ self.W2_ + self.b2_).ravel()

    def predict(self, X):
        """Predict 0/1 labels per row of X."""
        return (self.predict_proba(X) >= 0.5).astype(int)


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


if __name__ == "__main__":
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([0, 1, 1, 0])
    mlp = MLP(hidden=8, lr=0.5, epochs=5000, seed=0).fit(X_xor, y_xor)
    print("XOR accuracy:", round(accuracy(y_xor, mlp.predict(X_xor)), 3))
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (100, 2)), rng.normal(0, 1, (100, 2)) + [4, 0]])
    y = np.array([0] * 100 + [1] * 100)
    mlp2 = MLP(hidden=8, lr=0.5, epochs=1000, seed=0).fit(X, y)
    print("2D blobs accuracy:", round(accuracy(y, mlp2.predict(X)), 3))
