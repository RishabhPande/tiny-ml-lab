"""Hierarchical Clustering: agglomerative merging, bottom up

Agglomerative clustering starts with every point in its own
cluster and repeatedly merges the two closest clusters, until
either a requested number of clusters remains or the closest
pair is farther apart than a distance threshold. What closest
means is the linkage choice: single linkage uses the nearest
pair of points across the two clusters, complete linkage uses
the farthest pair, and average linkage uses the mean of all
cross pairs. The merge history is a dendrogram, so the same run
can be cut at different heights for different cluster counts.
This file keeps the simple version: a full pairwise distance
matrix and a direct search for the closest pair at each step,
which is fine at this scale (numpy only).
Run: python algorithms/hierarchical-clustering/impl.py
"""

import numpy as np


def make_blobs(seed=0, n_per=25):
    """Three round blobs, far apart relative to their spread.

    Centers (0, 0), (6, 0), and (3, 6), each with spread 0.4,
    so the nearest centers are about 15 spreads apart. Returns
    X and the true label of each point.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 6.0]])
    X = np.vstack([
        c + rng.normal(0.0, 0.4, (n_per, 2)) for c in centers
    ])
    y = np.repeat(np.arange(3), n_per)
    return X, y


def make_bridge(seed=0, n_per=20):
    """Two blobs joined by a thin chain of points.

    The blobs sit at (-3, 0) and (3, 0) with spread 0.3, about
    20 spreads apart, so on their own they are plainly two
    clusters. Between them, 13 bridge points are spaced about
    0.4 apart along the x axis, each step well under the gap
    between the blobs. Returns X and labels, where the bridge
    points carry label 2.
    """
    rng = np.random.default_rng(seed)
    left = rng.normal(0.0, 0.3, (n_per, 2)) + np.array([-3.0, 0.0])
    right = rng.normal(0.0, 0.3, (n_per, 2)) + np.array([3.0, 0.0])
    xs = np.linspace(-2.4, 2.4, 13)
    bridge = np.column_stack([xs, np.zeros(13)])
    bridge = bridge + rng.normal(0.0, 0.05, (13, 2))
    X = np.vstack([left, right, bridge])
    y = np.array([0] * n_per + [1] * n_per + [2] * 13)
    return X, y


def _pairwise_distances(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _linkage_distance(D, a, b, linkage):
    vals = D[np.ix_(a, b)]
    if linkage == "single":
        return float(np.min(vals))
    if linkage == "complete":
        return float(np.max(vals))
    return float(np.mean(vals))


class AgglomerativeClustering:
    """Agglomerative clustering with single, complete, or
    average linkage.

    Give exactly one stopping rule: n_clusters merges until
    that many clusters remain, and distance_threshold merges
    while the closest pair of clusters is within the threshold.
    Labels are assigned in the order the final clusters were
    formed, so the numbers themselves carry no meaning, only
    the grouping does.
    """

    def __init__(self, n_clusters=2, linkage="single",
                 distance_threshold=None):
        if linkage not in ("single", "complete", "average"):
            raise ValueError(
                "linkage must be single, complete, or average")
        if distance_threshold is not None and n_clusters != 2:
            raise ValueError(
                "give either n_clusters or distance_threshold, "
                "not both")
        if distance_threshold is None and n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if distance_threshold is not None and distance_threshold < 0:
            raise ValueError("distance_threshold must not be negative")
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.labels_ = None
        self.n_clusters_ = 0

    def fit(self, X):
        """Cluster the rows of X. Returns self."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty")
        if (self.distance_threshold is None
                and self.n_clusters > X.shape[0]):
            raise ValueError(
                "n_clusters must not exceed the number of points")
        D = _pairwise_distances(X)
        clusters = [[i] for i in range(X.shape[0])]
        while len(clusters) > 1:
            if (self.distance_threshold is None
                    and len(clusters) <= self.n_clusters):
                break
            best = None
            best_d = np.inf
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    d = _linkage_distance(
                        D, clusters[a], clusters[b], self.linkage)
                    if d < best_d:
                        best_d = d
                        best = (a, b)
            if (self.distance_threshold is not None
                    and best_d > self.distance_threshold):
                break
            a, b = best
            clusters[a] = clusters[a] + clusters[b]
            del clusters[b]
        labels = np.empty(X.shape[0], dtype=int)
        for k, members in enumerate(clusters):
            for i in members:
                labels[i] = k
        self.labels_ = labels
        self.n_clusters_ = len(clusters)
        return self

    def fit_predict(self, X):
        """Cluster X and return the label of each point."""
        return self.fit(X).labels_


if __name__ == "__main__":
    print("Hierarchical Clustering (seed 0)")

    # The works case: three far apart blobs, cut at three.
    X, y = make_blobs(seed=0)
    model = AgglomerativeClustering(
        n_clusters=3, linkage="single").fit(X)
    pure = all(
        len(np.unique(y[model.labels_ == k])) == 1
        for k in range(3))
    print(f"blobs: found {model.n_clusters_} clusters, "
          f"each one pure: {pure}")

    # The break case: a thin chain joins two far apart blobs.
    X, y = make_bridge(seed=0)
    clean = AgglomerativeClustering(
        distance_threshold=1.0, linkage="single").fit(X[:40])
    bridged = AgglomerativeClustering(
        distance_threshold=1.0, linkage="single").fit(X)
    print(f"threshold 1.0, no bridge: {clean.n_clusters_} clusters")
    print(f"threshold 1.0, with bridge: {bridged.n_clusters_} cluster")
    print("Single linkage only ever looks at the nearest pair, "
          "so a chain of small steps counts as contact. Each "
          "bridge step is about 0.4, well under the threshold, "
          "and the chain walks the merge across a gap of 6. "
          "The two blobs end up in one cluster even though no "
          "point in either blob is anywhere near the other.")
