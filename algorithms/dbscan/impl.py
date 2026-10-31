"""DBSCAN
Density-based clustering: points with enough close neighbors form a
cluster, the rest are noise (numpy only).
Run: python algorithms/dbscan/impl.py
"""

import numpy as np


class DBSCAN:
    """Density-based spatial clustering of applications with noise.

    A point is a core point when at least min_samples points (itself
    included) lie within eps of it. Core points linked by overlapping
    neighborhoods join one cluster. A non-core point within eps of a
    core point is a border point of that cluster. Every other point is
    noise, labeled -1. There is no predict for new points: DBSCAN
    clusters the data it is fitted on, it does not learn a boundary
    to apply elsewhere.
    """

    def __init__(self, eps=0.5, min_samples=5):
        if eps <= 0:
            raise ValueError("eps must be positive")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        self.eps = eps
        self.min_samples = min_samples

    def _neighbors(self, dist, i):
        """Indices within eps of point i, including i itself."""
        return np.where(dist[i] <= self.eps)[0]

    def fit(self, X):
        """Cluster X. Sets labels_, core_sample_indices_, n_clusters_."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        n = len(X)
        diff = X[:, None, :] - X[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        neighbor_lists = [self._neighbors(dist, i) for i in range(n)]
        is_core = np.array(
            [len(nb) >= self.min_samples for nb in neighbor_lists]
        )
        labels = np.full(n, -1, dtype=int)
        cluster_id = 0
        for i in range(n):
            if not is_core[i] or labels[i] != -1:
                continue
            labels[i] = cluster_id
            queue = [i]
            head = 0
            while head < len(queue):
                p = queue[head]
                head += 1
                for q in neighbor_lists[p]:
                    if labels[q] == -1:
                        labels[q] = cluster_id
                        if is_core[q]:
                            queue.append(q)
            cluster_id += 1
        self.labels_ = labels
        self.core_sample_indices_ = np.where(is_core)[0]
        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, X):
        """Fit on X and return the label of each point."""
        return self.fit(X).labels_


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    blob_a = rng.normal(0, 0.4, (80, 2))
    blob_b = rng.normal(0, 0.4, (80, 2)) + np.array([4.0, 4.0])
    noise = rng.uniform(-2.0, 6.0, (20, 2))
    X = np.vstack([blob_a, blob_b, noise])
    model = DBSCAN(eps=0.5, min_samples=5).fit(X)
    labels = model.labels_
    print("clusters found:", model.n_clusters_)
    for c in range(model.n_clusters_):
        print("cluster", c, "size:", int(np.sum(labels == c)))
    print("noise points:", int(np.sum(labels == -1)))
    print("note: the scattered points are noise, not a third cluster")
