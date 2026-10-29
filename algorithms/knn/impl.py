"""K-Nearest Neighbors
K-nearest neighbors classifier (numpy only).
Run: python algorithms/knn/impl.py
"""

import numpy as np


class KNNClassifier:
    """Majority vote of the k nearest training points. Ties go to the
    smallest label."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Memorize the training set. Returns self."""
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        return self

    def predict(self, X):
        """Predict one label per row of X."""
        X = np.asarray(X, dtype=float)
        dist = np.linalg.norm(X[:, None, :] - self.X_[None, :, :], axis=2)
        neighbors = np.argpartition(dist, self.n_neighbors, axis=1)[
            :, : self.n_neighbors
        ]
        out = []
        for row in neighbors:
            labels, counts = np.unique(self.y_[row], return_counts=True)
            out.append(labels[np.argmax(counts)])
        return np.array(out)


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_train = np.vstack(
        [rng.normal(0, 1, (100, 2)), rng.normal(0, 1, (100, 2)) + [4, 0]]
    )
    y_train = np.array([0] * 100 + [1] * 100)
    X_test = np.vstack(
        [rng.normal(0, 1, (50, 2)), rng.normal(0, 1, (50, 2)) + [4, 0]]
    )
    y_test = np.array([0] * 50 + [1] * 50)
    knn = KNNClassifier(n_neighbors=5).fit(X_train, y_train)
    print("test accuracy:", round(accuracy(y_test, knn.predict(X_test)), 3))
