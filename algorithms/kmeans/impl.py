"""K-Means
K-means clustering with Lloyd's algorithm (numpy only).
Run: python algorithms/kmeans/impl.py
"""

import numpy as np


class KMeans:
    """K-means with k-means++ init. fit sets centers_, labels_, inertia_."""

    def __init__(self, n_clusters=3, max_iter=100, seed=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed

    def _init_centers(self, X, rng):
        n = len(X)
        centers = [X[rng.integers(n)]]
        for _ in range(1, self.n_clusters):
            dist2 = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centers], axis=0
            )
            probs = dist2 / dist2.sum()
            centers.append(X[rng.choice(n, p=probs)])
        return np.array(centers)

    def fit(self, X):
        """Run Lloyd's algorithm. Returns self."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        centers = self._init_centers(X, rng)
        for _ in range(self.max_iter):
            dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dist, axis=1)
            new = np.array(
                [
                    X[labels == k].mean(axis=0)
                    if np.any(labels == k)
                    else centers[k]
                    for k in range(self.n_clusters)
                ]
            )
            if np.allclose(new, centers):
                break
            centers = new
        self.centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(np.sum(np.min(dist, axis=1) ** 2))
        return self

    def predict(self, X):
        """Assign each point to its nearest center."""
        X = np.asarray(X, dtype=float)
        dist = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2)
        return np.argmin(dist, axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true_centers = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.0]])
    X = np.vstack(
        [c + rng.normal(0, 0.7, (60, 2)) for c in true_centers]
    )
    km = KMeans(n_clusters=3, seed=0).fit(X)
    print("found centers:\n", np.round(km.centers_, 2))
    print("inertia:", round(km.inertia_, 1))
