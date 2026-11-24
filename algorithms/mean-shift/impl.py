"""Mean Shift: climb to the nearest dense spot, and call the
spots you find clusters.

Mean shift needs no cluster count up front. Start a walk from
every point: look at the neighbours inside a window set by the
bandwidth, move to their weighted centre, and repeat until the
walk stops moving. Points whose walks end at the same peak
belong to the same cluster, so the number of peaks is the
number of clusters the data gets. A Gaussian kernel weights
nearby points more than distant ones, which makes the walks
smooth. The whole method hangs on the bandwidth. Too small and
every bump in the noise becomes its own peak; too large and
the window covers two real clusters at once, their walks climb
to one shared peak in between, and two clusters come back as
one (numpy only).
Run: python algorithms/mean-shift/impl.py
"""

import numpy as np


def make_blobs(seed=0, n_per_blob=60, separation=6.0, spread=0.6):
    """Two round blobs, far apart, with known labels.

    Centres sit at (0, 0) and (separation, 0). Each blob is
    Gaussian noise with the given spread around its centre.
    Returns (X, y, centres), so the caller knows the truth the
    clustering has to recover.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [separation, 0.0]])
    X0 = centres[0] + rng.normal(scale=spread, size=(n_per_blob, 2))
    X1 = centres[1] + rng.normal(scale=spread, size=(n_per_blob, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_blob + [1] * n_per_blob)
    return X, y, centres


def cluster_accuracy(y_true, labels):
    """Best match accuracy over the two label orders.

    Cluster ids are arbitrary, so the score tries both the
    identity match and the swapped match and keeps the better
    one. Only defined for two true classes.
    """
    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    direct = float(np.mean(y_true == labels))
    swapped = float(np.mean(y_true == (1 - labels)))
    return max(direct, swapped)


class MeanShift:
    """Mean shift clustering with a Gaussian kernel.

    fit walks every point uphill: at each step the new position
    is the kernel weighted mean of all points, with weights
    exp(-distance squared / (2 * bandwidth squared)). A walk
    stops when a step moves less than tol, or after max_iter
    steps. Endpoints closer than half the bandwidth merge into
    one centre. self.cluster_centers_ holds the merged peaks
    and self.labels_ the nearest centre index for each point.
    """

    def __init__(self, bandwidth=2.0, seed=0, max_iter=100, tol=1e-3):
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self.bandwidth = bandwidth
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol

    def _shift(self, X, start):
        point = np.array(start, dtype=float)
        for _ in range(self.max_iter):
            dist_sq = np.sum((X - point) ** 2, axis=1)
            weights = np.exp(-dist_sq / (2.0 * self.bandwidth ** 2))
            new_point = (weights @ X) / weights.sum()
            if np.linalg.norm(new_point - point) < self.tol:
                point = new_point
                break
            point = new_point
        return point

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        endpoints = np.array([self._shift(X, x) for x in X])
        centres = []
        for end in endpoints:
            placed = False
            for centre in centres:
                if np.linalg.norm(end - centre) < 0.5 * self.bandwidth:
                    placed = True
                    break
            if not placed:
                centres.append(end)
        self.cluster_centers_ = np.array(centres)
        dists = np.linalg.norm(
            X[:, None, :] - self.cluster_centers_[None, :, :], axis=2
        )
        self.labels_ = np.argmin(dists, axis=1)
        self.n_clusters_ = len(centres)
        return self

    def fit_predict(self, X):
        """Fit on X and return the cluster label of each point."""
        self.fit(X)
        return self.labels_


if __name__ == "__main__":
    X, y, centres = make_blobs(seed=0)

    print("Mean shift with bandwidth 2.0 on two blobs (seed 0)")
    model = MeanShift(bandwidth=2.0, seed=0)
    labels = model.fit_predict(X)
    print(f"clusters found: {model.n_clusters_}")
    print(f"cluster accuracy: {cluster_accuracy(y, labels):.3f}")
    print("with a window matched to the blob size, the walks end")
    print("at two peaks, one per blob")

    print("\nMean shift with bandwidth 8.0 on the same blobs (seed 0)")
    wide = MeanShift(bandwidth=8.0, seed=0)
    wide_labels = wide.fit_predict(X)
    print(f"clusters found: {wide.n_clusters_}")
    print("a window wider than the gap between the blobs covers")
    print("both at once, so every walk climbs to one shared peak")
    print("and two clusters come back as one")
