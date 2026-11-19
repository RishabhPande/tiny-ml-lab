"""t-SNE: a map that keeps neighbors close and lies about distance

t-SNE turns high dimensional points into a two dimensional map.
For each point it builds a probability distribution over its
neighbors, with a Gaussian kernel whose width is tuned per point
so that every point has the same effective number of neighbors,
called the perplexity. It then places points on the map so that
a similar neighbor distribution, this time with a heavy tailed
Student t kernel, matches the first one as closely as possible,
measured by KL divergence and fitted by gradient descent with
momentum and early exaggeration. Neighbors stay neighbors, so
clusters you can see on the map are real. What the map does not
keep is global geometry: the distance between two clusters on
the map, and the size of a cluster on the map, carry almost no
information about the original data (numpy only).
Run: python algorithms/tsne/impl.py
"""

import numpy as np


def make_blobs(seed=0, n_per=30, dim=8):
    """Three round blobs in 8 dimensions, spread 0.5.

    Centers are drawn once with spread 4, so the blobs sit many
    spreads apart and each blob is a tight local neighborhood.
    Returns X and the true label of each point.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 4.0, (3, dim))
    parts = []
    labels = []
    for k in range(3):
        parts.append(centers[k] + rng.normal(0.0, 0.5, (n_per, dim)))
        labels.extend([k] * n_per)
    return np.vstack(parts), np.array(labels)


def make_uneven_blobs(seed=0, n_per=30):
    """Three blobs in a line: A at 0, B at 3, C at 30.

    All three have the same spread, 0.5, so the only structure
    beyond the clusters themselves is the spacing: C is ten
    times farther from A than B is. The break-it case shows the
    map throwing that spacing away. Returns X and labels.
    """
    rng = np.random.default_rng(seed)
    parts = []
    labels = []
    for k, center in enumerate((0.0, 3.0, 30.0)):
        pts = rng.normal(0.0, 0.5, (n_per, 4))
        pts[:, 0] += center
        parts.append(pts)
        labels.extend([k] * n_per)
    return np.vstack(parts), np.array(labels)


def _sq_dists(X):
    sq = np.sum(X ** 2, axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def _conditional_p(D, perplexity, tol=1e-5, max_iter=50):
    """Neighbor probabilities with per point widths.

    For each point, binary search the Gaussian width so the
    entropy of its neighbor distribution equals log2 of the
    perplexity. Returns the conditional probability matrix.
    """
    n = D.shape[0]
    target = np.log2(perplexity)
    P = np.zeros((n, n))
    for i in range(n):
        Di = np.delete(D[i], i)
        beta = 1.0
        beta_min = -np.inf
        beta_max = np.inf
        Pi = np.zeros(n - 1)
        for _ in range(max_iter):
            expd = np.exp(-Di * beta)
            total = expd.sum()
            if total == 0.0:
                break
            Pi = expd / total
            entropy = (np.log(total)
                       + beta * float(np.sum(Di * expd)) / total)
            entropy = entropy / np.log(2.0)
            diff = entropy - target
            if abs(diff) < tol:
                break
            if diff > 0.0:
                beta_min = beta
                beta = (beta * 2.0 if beta_max == np.inf
                        else (beta + beta_max) / 2.0)
            else:
                beta_max = beta
                beta = (beta / 2.0 if beta_min == -np.inf
                        else (beta + beta_min) / 2.0)
        P[i, np.arange(n) != i] = Pi
    return P


class TSNE:
    """t-SNE embedding into 2 dimensions, fitted by gradient descent.

    perplexity is the effective number of neighbors each point
    pays attention to and must be smaller than the number of
    points. The fit uses early exaggeration for the first
    exaggeration_iters steps, then plain probabilities, with
    per coordinate adaptive gains and momentum that rises from
    0.5 to 0.8 partway through, the standard recipe.
    """

    def __init__(self, perplexity=20.0, n_iter=500, learning_rate=200.0,
                 exaggeration=12.0, exaggeration_iters=100, seed=0):
        if perplexity < 1.0:
            raise ValueError("perplexity must be at least 1")
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.perplexity = float(perplexity)
        self.n_iter = int(n_iter)
        self.learning_rate = float(learning_rate)
        self.exaggeration = float(exaggeration)
        self.exaggeration_iters = int(exaggeration_iters)
        self.seed = seed
        self.embedding_ = None

    def fit_transform(self, X):
        """Embed the rows of X into 2D. Returns the embedding."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        n = X.shape[0]
        if n < 3:
            raise ValueError("X must have at least 3 points")
        if self.perplexity >= n:
            raise ValueError(
                "perplexity must be smaller than the number of points")
        rng = np.random.default_rng(self.seed)
        P = _conditional_p(_sq_dists(X), self.perplexity)
        P = (P + P.T) / (2.0 * n)
        P = np.maximum(P, 1e-12)
        Y = rng.normal(0.0, 1e-4, (n, 2))
        velocity = np.zeros_like(Y)
        gains = np.ones_like(Y)
        for it in range(self.n_iter):
            num = 1.0 / (1.0 + _sq_dists(Y))
            np.fill_diagonal(num, 0.0)
            Q = np.maximum(num / num.sum(), 1e-12)
            target = (P * self.exaggeration
                      if it < self.exaggeration_iters else P)
            diff = Y[:, None, :] - Y[None, :, :]
            grad = 4.0 * np.sum(
                ((target - Q) * num)[:, :, None] * diff, axis=1)
            momentum = 0.5 if it < 250 else 0.8
            gains = np.where(np.sign(grad) != np.sign(velocity),
                             gains + 0.2, gains * 0.8)
            gains = np.maximum(gains, 0.01)
            velocity = momentum * velocity - self.learning_rate * (
                gains * grad)
            Y = Y + velocity
            Y = Y - Y.mean(axis=0)
        self.embedding_ = Y
        return Y

    def fit(self, X):
        """Embed X and store the result. Returns self."""
        self.fit_transform(X)
        return self


def _knn_accuracy(Y, y, k=5):
    """Leave one out k nearest neighbor label accuracy in Y."""
    D = _sq_dists(Y)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    n_classes = int(y.max()) + 1
    pred = np.array([
        np.bincount(y[row], minlength=n_classes).argmax() for row in idx])
    return float(np.mean(pred == y))


def _centroid_distance_ratio(X, y):
    """Distance from cluster 0 to cluster 2 over 0 to cluster 1."""
    cents = np.array([X[y == k].mean(axis=0) for k in range(3)])
    far = np.linalg.norm(cents[0] - cents[2])
    near = np.linalg.norm(cents[0] - cents[1])
    return float(far / near)


if __name__ == "__main__":
    print("t-SNE (seed 0)")

    # The works case: local neighborhoods survive the map.
    X, y = make_blobs(seed=0)
    Y = TSNE().fit_transform(X)
    print(f"blobs: 5-NN accuracy in the map "
          f"{_knn_accuracy(Y, y):.3f} (1.000 in the raw data too)")

    # The break case: the spacing between clusters does not.
    X, y = make_uneven_blobs(seed=0)
    Y = TSNE().fit_transform(X)
    print(f"uneven blobs: centroid distance ratio far/near "
          f"{_centroid_distance_ratio(X, y):.2f} in the raw data, "
          f"{_centroid_distance_ratio(Y, y):.2f} on the map")
    print("The map keeps who is next to whom and throws away how "
          "far apart the groups are. Cluster A and B sit 3 units "
          "apart and C sits 30 units away, a ratio near 10, yet "
          "on the map all three clusters end up about equally "
          "spaced, a ratio near 1. Reading distances or cluster "
          "sizes off a t-SNE plot is reading numbers the method "
          "never tried to keep.")
