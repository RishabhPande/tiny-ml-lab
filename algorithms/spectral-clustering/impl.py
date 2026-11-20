"""Spectral Clustering: cluster the eigenvectors, not the points

Plain distance clustering fails on shapes like two concentric
rings, because the middle of the inner ring and the middle of
the outer ring are the same point, so no center based method
can separate them. Spectral clustering changes the question.
Build a similarity graph first: points are nodes, and an edge
weight says how alike two points are, here a Gaussian kernel
with width sigma. Then take the graph Laplacian, a matrix that
records how the graph is connected, and compute its smallest
eigenvectors. Points that are strongly connected inside the
graph land close together in that eigenvector space even when
they are far apart in the raw space, so a plain k-means on the
eigenvector rows pulls the rings apart. The width sigma is the
whole game: too small and the graph falls apart into isolated
points, too large and everything looks alike (numpy only).
Run: python algorithms/spectral-clustering/impl.py
"""

import numpy as np


def make_rings(seed=0, n_per=60):
    """Two concentric rings, radii 1 and 3, noise 0.08.

    The inner ring has label 0 and the outer ring label 1.
    Both rings share the same center, so no method that assigns
    points to centers can split them: the best a center can do
    is cut the picture with a straight line. Returns X and the
    true label of each point.
    """
    rng = np.random.default_rng(seed)
    parts = []
    labels = []
    for k, radius in enumerate((1.0, 3.0)):
        theta = rng.uniform(0.0, 2.0 * np.pi, n_per)
        pts = np.column_stack(
            [radius * np.cos(theta), radius * np.sin(theta)])
        pts = pts + rng.normal(0.0, 0.08, (n_per, 2))
        parts.append(pts)
        labels.extend([k] * n_per)
    return np.vstack(parts), np.array(labels)


def make_blobs(seed=0, n_per=25):
    """Two round blobs at (-2, 0) and (2, 0), spread 0.4.

    The centers are 10 spreads apart, so with a sane sigma the
    graph has two clear pieces and the method recovers the
    blobs exactly. The break-it case reuses these blobs with a
    sigma so small that almost no edge survives. Returns X and
    the true label of each point.
    """
    rng = np.random.default_rng(seed)
    left = rng.normal(0.0, 0.4, (n_per, 2)) + np.array([-2.0, 0.0])
    right = rng.normal(0.0, 0.4, (n_per, 2)) + np.array([2.0, 0.0])
    X = np.vstack([left, right])
    y = np.array([0] * n_per + [1] * n_per)
    return X, y


def _affinity(X, sigma):
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    W = np.exp(-d2 / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    return W


def _kmeans(Z, k, seed=0, n_init=10, max_iter=100):
    """Plain Lloyd k-means with seeded restarts.

    Keeps the best run by inertia, so the result is fixed for a
    fixed seed. Used on the eigenvector rows, not the raw data.
    """
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    best_labels = None
    best_inertia = np.inf
    for _ in range(n_init):
        centers = Z[rng.choice(n, size=k, replace=False)]
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            d2 = np.sum((Z[:, None, :] - centers[None, :, :]) ** 2,
                        axis=2)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for j in range(k):
                if np.any(labels == j):
                    centers[j] = Z[labels == j].mean(axis=0)
        inertia = float(np.sum((Z - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels


class SpectralClustering:
    """Spectral clustering with a Gaussian kernel graph.

    sigma is the kernel width: pairs much closer than sigma get
    weight near 1, pairs much farther get weight near 0. The
    normalized Laplacian's smallest n_clusters eigenvectors
    form the embedding, rows are rescaled to unit length, and
    k-means on those rows gives the labels. Label numbers carry
    no meaning, only the grouping does.
    """

    def __init__(self, n_clusters=2, sigma=0.5, seed=0):
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.seed = seed
        self.labels_ = None
        self.embedding_ = None

    def fit(self, X):
        """Cluster the rows of X. Returns self."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty")
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                "n_clusters must not exceed the number of points")
        W = _affinity(X, self.sigma)
        degree = W.sum(axis=1)
        # A point with no surviving edges has degree 0; leave its
        # inverse root at 0 so it stays out of the embedding
        # instead of blowing it up.
        inv_sqrt = np.where(degree > 0.0,
                            1.0 / np.sqrt(np.maximum(degree, 1e-300)),
                            0.0)
        S = inv_sqrt[:, None] * W * inv_sqrt[None, :]
        L = np.eye(X.shape[0]) - S
        _, vecs = np.linalg.eigh(L)
        Z = vecs[:, : self.n_clusters]
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        Z = np.where(norms > 0.0, Z / np.maximum(norms, 1e-300), Z)
        self.embedding_ = Z
        self.labels_ = _kmeans(Z, self.n_clusters, seed=self.seed)
        return self

    def fit_predict(self, X):
        """Cluster X and return the label of each point."""
        return self.fit(X).labels_


def _best_accuracy(y_true, y_pred, k):
    """Accuracy after the best relabeling of the predictions.

    Tries both orders for k = 2, which is all this file needs:
    label numbers are arbitrary, so a perfect grouping can
    still score 0 if the numbers happen to be flipped.
    """
    if k == 2:
        plain = np.mean(y_true == y_pred)
        flipped = np.mean(y_true == (1 - y_pred))
        return max(plain, flipped)
    return float(np.mean(y_true == y_pred))


if __name__ == "__main__":
    print("Spectral Clustering (seed 0)")

    # The works case: two rings, one shared center.
    X, y = make_rings(seed=0)
    model = SpectralClustering(n_clusters=2).fit(X)
    acc = _best_accuracy(y, model.labels_, 2)
    print(f"rings: best-label accuracy {acc:.3f}")

    # The break case: same code, sane blobs, sigma far too small.
    X, y = make_blobs(seed=0)
    good = SpectralClustering(n_clusters=2).fit(X)
    bad_sigma = 0.0001
    bad = SpectralClustering(n_clusters=2, sigma=bad_sigma).fit(X)
    print(f"blobs, sigma 0.5: accuracy "
          f"{_best_accuracy(y, good.labels_, 2):.3f}")
    print(f"blobs, sigma {bad_sigma}: accuracy "
          f"{_best_accuracy(y, bad.labels_, 2):.3f}")
    surviving = np.count_nonzero(_affinity(X, bad_sigma))
    print(f"blobs, sigma {bad_sigma}: surviving directed edges {surviving}")
    print("At this bandwidth every off-diagonal kernel weight "
          "underflows to zero. The graph has no edges and its "
          "normalized Laplacian is the identity: all eigenvalues "
          "are equal. The returned grouping depends on the chosen "
          "eigenbasis and input order, not on blob structure. "
          "Accuracy can look better than chance by accident; "
          "the missing edges are the failure to inspect.")
