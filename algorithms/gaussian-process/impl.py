"""Gaussian Process Regression: fit a distribution over curves,
not just one curve, and read the uncertainty off it.

A Gaussian process regressor never commits to a formula for the
curve. Instead it says: points close together should have
similar values, with closeness measured by a kernel. Given
training points, the prediction at a new point is a weighted
blend of the training values, where the weights come from
solving one linear system built from the kernel, and the same
math yields a standard deviation that is small near the data
and large far from it. The length scale inside the kernel is
the whole game. Matched to the data it lets the curve bend at
the right rate. Set it far wider than the range of the data and
every pair of points looks equally close, the kernel matrix
goes nearly constant, and the prediction flattens toward the
training mean no matter what the data does (numpy only).
Run: python algorithms/gaussian-process/impl.py
"""

import numpy as np


def make_wave(seed=0, n_train=24, noise=0.1):
    """A noisy sine wave on a fixed grid, with a clean test grid.

    Training inputs are n_train evenly spaced points in [0, 6],
    targets are sin(2x) plus Gaussian noise with the given
    scale, so the wave completes about two full cycles. The
    test grid is 61 clean points over the same range, with the
    noiseless sin(2x) values as targets. Returns
    (X_train, y_train, X_test, y_test), inputs as column arrays.
    """
    rng = np.random.default_rng(seed)
    X_train = np.linspace(0.0, 6.0, n_train).reshape(-1, 1)
    y_train = np.sin(2.0 * X_train).ravel() + rng.normal(scale=noise, size=n_train)
    X_test = np.linspace(0.0, 6.0, 61).reshape(-1, 1)
    y_test = np.sin(2.0 * X_test).ravel()
    return X_train, y_train, X_test, y_test


def r2_score(y_true, y_pred):
    """Coefficient of determination against the mean baseline."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def rbf_kernel(X1, X2, length_scale=1.0, signal_std=1.0):
    """Squared exponential kernel between two sets of points.

    k(x, z) = signal_std squared times
    exp(-distance squared / (2 * length_scale squared)).
    Returns the (len(X1), len(X2)) matrix of pairwise values.
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True).T
    dist_sq = np.maximum(sq1 + sq2 - 2.0 * (X1 @ X2.T), 0.0)
    return signal_std ** 2 * np.exp(-dist_sq / (2.0 * length_scale ** 2))


class GaussianProcessRegressor:
    """Gaussian process regression with an RBF kernel.

    fit stores the training data, adds the noise variance to the
    kernel diagonal, and factorises the result with a Cholesky
    decomposition (a small jitter keeps the factorisation
    stable). predict solves against that factor to return the
    posterior mean and standard deviation at each test point.
    """

    def __init__(self, length_scale=1.0, signal_std=1.0, noise_std=0.1):
        if length_scale <= 0:
            raise ValueError("length_scale must be positive")
        if signal_std <= 0:
            raise ValueError("signal_std must be positive")
        if noise_std < 0:
            raise ValueError("noise_std must not be negative")
        self.length_scale = length_scale
        self.signal_std = signal_std
        self.noise_std = noise_std

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.X_train_ = X
        self.y_train_ = y
        K = rbf_kernel(X, X, self.length_scale, self.signal_std)
        K[np.diag_indices_from(K)] += self.noise_std ** 2 + 1e-10
        self.L_ = np.linalg.cholesky(K)
        self.alpha_ = np.linalg.solve(
            self.L_.T, np.linalg.solve(self.L_, y)
        )
        return self

    def predict(self, X):
        """Return (mean, std) of the prediction at each row of X."""
        X = np.asarray(X, dtype=float)
        K_star = rbf_kernel(
            self.X_train_, X, self.length_scale, self.signal_std
        )
        mean = K_star.T @ self.alpha_
        V = np.linalg.solve(self.L_, K_star)
        var = self.signal_std ** 2 - np.sum(V ** 2, axis=0)
        std = np.sqrt(np.maximum(var, 0.0))
        return mean, std


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = make_wave(seed=0)

    print("Gaussian process with length scale 1.0 on a sine wave (seed 0)")
    model = GaussianProcessRegressor(length_scale=1.0, noise_std=0.1)
    model.fit(X_train, y_train)
    mean, std = model.predict(X_test)
    print(f"test R2: {r2_score(y_test, mean):.3f}")
    far_mean, far_std = model.predict(np.array([[12.0]]))
    print(f"std inside the data range: {std[30]:.3f}")
    print(f"std at x = 12, far outside it: {far_std[0]:.3f}")
    print("near the data the process is sure of itself; far away")
    print("the error bars grow back toward the prior")

    print("\nGaussian process with length scale 10.0 on the same wave (seed 0)")
    wide = GaussianProcessRegressor(length_scale=10.0, noise_std=0.1)
    wide.fit(X_train, y_train)
    wide_mean, wide_std = wide.predict(X_test)
    print(f"test R2: {r2_score(y_test, wide_mean):.3f}")
    print("a length scale wider than the whole data range makes")
    print("every pair of points look equally close, so the curve")
    print("flattens toward the training mean and misses the wave")
