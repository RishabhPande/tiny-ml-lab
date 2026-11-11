"""Isolation Forest: anomaly detection by random isolation.

Normal points live in dense regions, so a random axis-aligned
split rarely separates one of them from all the others. An odd
point out on its own gets cut off from the crowd in just a few
splits. An isolation tree makes that idea literal: pick a random
feature, pick a random split between its min and max in the
current node, and recurse until every point sits alone or a
depth limit is hit. Average the path length over many trees and
short paths mean anomalies. The score rescales that average by
the expected path length for ordinary data,

    score = 2 ** (-average_path / c(sample_size))

so a score near 1 is very anomalous and a score near 0.5 is
unremarkable.

The break-it test shows masking. If the anomalies arrive as one
tight cluster instead of scattered lone points, the cluster
protects its members: the first splits separate the whole
cluster from the normal data, but then each member still needs
several more splits to be isolated from its neighbours. A big
enough cluster scores as ordinary as the normal data, even
though it sits just as far away as scattered anomalies that
the forest catches almost perfectly.

Run: python algorithms/isolation-forest/impl.py
"""

import numpy as np


def expected_path_length(n):
    """Average path length of an unsuccessful search in a binary
    search tree of n points, the c(n) normalizer from the paper.

    c(1) = 0, c(2) = 1, and for larger n it is
    2 * H(n - 1) - 2 * (n - 1) / n, where H is the harmonic
    number, approximated here by ln(n) plus the Euler constant.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    harmonic = np.log(n - 1) + 0.5772156649
    return float(2 * harmonic - 2 * (n - 1) / n)


class _Node:
    """One node of an isolation tree. A leaf has feature None
    and only remembers how many points reached it, which the
    path length needs for its leaf-size correction."""

    __slots__ = ("feature", "split", "left", "right", "size")

    def __init__(self, size, feature=None, split=None,
                 left=None, right=None):
        self.size = size
        self.feature = feature
        self.split = split
        self.left = left
        self.right = right


class IsolationForest:
    """A forest of random isolation trees.

    Each tree is built on a random subsample of the training
    data without replacement. `sample_size` is that subsample
    size (capped at the number of training points) and the
    depth limit is ceil(log2(sample_size)), the usual choice
    from the paper: deeper than that, the trees would only be
    separating ordinary points from each other.
    """

    def __init__(self, n_trees=100, sample_size=64, seed=0):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.seed = seed
        self.trees = []
        self.psi = sample_size

    def _build(self, x, depth, max_depth, rng):
        n = len(x)
        if n <= 1 or depth >= max_depth:
            return _Node(n)
        feature = int(rng.integers(x.shape[1]))
        low = float(x[:, feature].min())
        high = float(x[:, feature].max())
        if low == high:
            return _Node(n)
        split = float(rng.uniform(low, high))
        goes_left = x[:, feature] < split
        left = x[goes_left]
        right = x[~goes_left]
        if len(left) == 0 or len(right) == 0:
            return _Node(n)
        return _Node(n, feature, split,
                     self._build(left, depth + 1, max_depth, rng),
                     self._build(right, depth + 1, max_depth, rng))

    def fit(self, x):
        """Build every tree on its own subsample. Returns self."""
        x = np.asarray(x, dtype=float)
        n = len(x)
        self.psi = min(self.sample_size, n)
        max_depth = int(np.ceil(np.log2(self.psi))) if self.psi > 1 else 0
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.choice(n, size=self.psi, replace=False)
            self.trees.append(self._build(x[idx], 0, max_depth, rng))
        return self

    def _path_length(self, point, node, depth=0):
        if node.feature is None:
            return depth + expected_path_length(node.size)
        if point[node.feature] < node.split:
            return self._path_length(point, node.left, depth + 1)
        return self._path_length(point, node.right, depth + 1)

    def average_path_length(self, x):
        """Mean path length over all trees, one value per row."""
        x = np.asarray(x, dtype=float)
        paths = np.array([
            [self._path_length(point, tree) for tree in self.trees]
            for point in x
        ])
        return paths.mean(axis=1)

    def anomaly_score(self, x):
        """Score in (0, 1) per row, higher means more anomalous.

        A point whose average path equals the expected path for
        ordinary data scores 0.5. Much shorter paths push the
        score toward 1, much longer paths toward 0.
        """
        average = self.average_path_length(x)
        return 2 ** (-average / expected_path_length(self.psi))


def make_data(seed=0):
    """Synthetic demo data: 200 normal points from a standard
    normal blob, plus 5 lone outliers near the corners, far from
    the blob. Returns (x, y) with y = 1 marking the outliers."""
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, (200, 2))
    outliers = np.array([[5.0, 5.0], [-5.0, 5.0], [5.0, -5.0],
                         [-5.0, -5.0], [6.0, 0.0]])
    x = np.vstack([normal, outliers])
    y = np.array([0] * 200 + [1] * 5)
    return x, y


def make_clustered_data(seed=3):
    """The break case: 150 normal points, plus anomalies at the
    same distance from the blob in two arrangements. Returns
    (x_scattered, x_clustered), each with the normal points in
    the first 150 rows and 100 anomalies after them. Scattered
    anomalies sit alone on a ring of radius 7; clustered ones
    form one tight blob at (5, 5) with standard deviation 0.05."""
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, (150, 2))
    scattered = rng.normal(0.0, 1.0, (100, 2))
    norms = np.linalg.norm(scattered, axis=1, keepdims=True)
    scattered = scattered / norms * 7.0
    clustered = rng.normal(0.0, 0.05, (100, 2)) + np.array([5.0, 5.0])
    return (np.vstack([normal, scattered]),
            np.vstack([normal, clustered]))


def recall_at_k(scores, labels, k):
    """Fraction of the true anomalies among the k highest scores,
    counting a hit only when the flagged point is an anomaly.
    Returns hits divided by the number of true anomalies, the
    usual recall when exactly k points are flagged."""
    flagged = np.argsort(-scores)[:k]
    hits = int(np.sum(labels[flagged] == 1))
    return hits / float(np.sum(labels == 1))


if __name__ == "__main__":
    x, y = make_data()
    forest = IsolationForest().fit(x)
    scores = forest.anomaly_score(x)
    print("Isolation Forest on a normal blob plus 5 lone outliers")
    print(f"Mean score, normal points: {scores[y == 0].mean():.3f}")
    print(f"Scores, the 5 outliers: "
          f"{np.round(scores[y == 1], 3).tolist()}")
    print(f"All 5 outliers are the 5 highest scores: "
          f"{recall_at_k(scores, y, 5) == 1.0}")

    x_scattered, x_clustered = make_clustered_data()
    y_big = np.array([0] * 150 + [1] * 100)
    scattered_scores = IsolationForest().fit(
        x_scattered).anomaly_score(x_scattered)
    clustered_scores = IsolationForest().fit(
        x_clustered).anomaly_score(x_clustered)
    print("\nMasking: 100 anomalies, scattered versus one tight cluster")
    print(f"Scattered: mean anomaly score "
          f"{scattered_scores[y_big == 1].mean():.3f}, "
          f"recall when flagging 100: "
          f"{recall_at_k(scattered_scores, y_big, 100):.2f}")
    print(f"Clustered: mean anomaly score "
          f"{clustered_scores[y_big == 1].mean():.3f}, "
          f"recall when flagging 100: "
          f"{recall_at_k(clustered_scores, y_big, 100):.2f}")
    print("Same distance from the blob, but the cluster hides its")
    print("members from each other, so most of them score as normal.")
