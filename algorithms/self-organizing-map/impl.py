"""Self-Organizing Map: a grid of prototypes that learns to
spread itself over the data while keeping neighbours on the
grid near each other in the data space.

Each grid unit holds a prototype vector. Training repeats one
simple step: find the unit whose prototype is closest to a
data point (the best matching unit, or BMU), then pull that
unit and its grid neighbours toward the point, the BMU most
and the neighbours less, by an amount that shrinks with grid
distance. The pull strength and the neighbourhood radius both
decay over time, so the map first unfolds roughly and then
settles into fine detail. Because neighbours move together,
the finished map is topology preserving: points that are close
in the data tend to land on the same unit or on neighbouring
units, which is what makes the grid useful as a 2D picture of
high dimensional data. The catch is capacity: a grid with too
few units cannot give distinct clusters their own territory,
and a 1 by 1 grid collapses every point onto a single
prototype no matter how the training runs (numpy only).
Run: python algorithms/self-organizing-map/impl.py
"""

import numpy as np


def make_blob_data(seed=0, n_per_blob=120, blob_std=0.4):
    """Three well separated 2D blobs.

    Centres sit at (0, 0), (4, 0) and (2, 3.5), far apart
    compared with the per-blob spread. Returns (X, labels)
    with points grouped by blob in order.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5]])
    blocks = [
        centre + rng.normal(scale=blob_std, size=(n_per_blob, 2))
        for centre in centres
    ]
    X = np.vstack(blocks)
    labels = np.repeat(np.arange(3), n_per_blob)
    return X, labels


class SelfOrganizingMap:
    """A Kohonen self-organizing map trained by BMU updates.

    The prototypes start as random data points, so the map
    begins inside the data cloud. Each step draws one point,
    finds its BMU, and moves every unit toward the point with
    a Gaussian weight over grid distance from the BMU, scaled
    by the current learning rate. The learning rate and the
    radius decay exponentially from their starting values to
    5 percent of them across the run, so late steps only polish.
    """

    def __init__(self, grid_rows=8, grid_cols=8, learning_rate=0.5,
                 radius=None, n_iter=4000):
        if grid_rows < 1 or grid_cols < 1:
            raise ValueError("the grid needs at least one row and one column")
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.learning_rate = learning_rate
        self.radius = radius if radius is not None else max(grid_rows, grid_cols) / 2.0
        self.n_iter = n_iter
        rows, cols = np.meshgrid(
            np.arange(grid_rows), np.arange(grid_cols), indexing="ij"
        )
        self._grid_positions = np.stack([rows.ravel(), cols.ravel()], axis=1)

    def fit(self, X, seed=0):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        rng = np.random.default_rng(seed)
        n, d = X.shape
        n_units = self.grid_rows * self.grid_cols
        start = X[rng.choice(n, size=n_units, replace=n_units > n)]
        weights = start + rng.normal(scale=0.01, size=(n_units, d))

        for step in range(self.n_iter):
            frac = step / max(self.n_iter - 1, 1)
            lr = max(self.learning_rate * np.exp(-3.0 * frac), 1e-3)
            sigma = max(self.radius * np.exp(-3.0 * frac), 0.3)
            point = X[rng.integers(n)]
            dists = np.sum((weights - point) ** 2, axis=1)
            bmu = int(np.argmin(dists))
            grid_dists = np.sum(
                (self._grid_positions - self._grid_positions[bmu]) ** 2, axis=1
            )
            influence = np.exp(-grid_dists / (2.0 * sigma ** 2))
            weights += lr * influence[:, None] * (point - weights)

        self.weights_ = weights.reshape(self.grid_rows, self.grid_cols, d)
        return self

    def bmu(self, X):
        """Grid (row, col) of the closest prototype for each row of X."""
        X = np.asarray(X, dtype=float)
        flat = self.weights_.reshape(-1, self.weights_.shape[-1])
        dists = np.sum((X[:, None, :] - flat[None, :, :]) ** 2, axis=2)
        idx = np.argmin(dists, axis=1)
        return np.stack(
            [idx // self.grid_cols, idx % self.grid_cols], axis=1
        )

    def quantization_error(self, X):
        """Mean distance from each point to its closest prototype."""
        X = np.asarray(X, dtype=float)
        flat = self.weights_.reshape(-1, self.weights_.shape[-1])
        dists = np.sqrt(np.sum((X[:, None, :] - flat[None, :, :]) ** 2, axis=2))
        return float(np.mean(np.min(dists, axis=1)))


if __name__ == "__main__":
    X, labels = make_blob_data(seed=0)

    print("Self-organizing map, 8 by 8 grid, on three blobs (seed 0)")
    som = SelfOrganizingMap(grid_rows=8, grid_cols=8, n_iter=4000)
    som.fit(X, seed=0)
    print(f"quantization error: {som.quantization_error(X):.3f}")
    units = som.bmu(X)
    modes = []
    for blob in range(3):
        rows, counts = np.unique(
            units[labels == blob, 0] * 8 + units[labels == blob, 1],
            return_counts=True,
        )
        modes.append(rows[np.argmax(counts)])
    mode_cells = [(int(m) // 8, int(m) % 8) for m in modes]
    print(f"busiest unit per blob: {mode_cells}")
    print("each blob claims its own busiest unit, far from the")
    print("others, so the grid has unfolded into a map of the data")

    print("\nSelf-organizing map, 1 by 1 grid, on the same data (seed 0)")
    tiny = SelfOrganizingMap(grid_rows=1, grid_cols=1, n_iter=4000)
    tiny.fit(X, seed=0)
    print(f"quantization error: {tiny.quantization_error(X):.3f}")
    print("one unit has to stand in for three far apart blobs, so")
    print("every point sits far from the only prototype there is")
