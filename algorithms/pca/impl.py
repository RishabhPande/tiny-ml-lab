"""PCA
Principal component analysis via eigendecomposition (numpy only).
Run: python algorithms/pca/impl.py
"""

import numpy as np


class PCA:
    """PCA with n_components principal axes, sorted by variance."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        """Center X and eigendecompose its covariance. Returns self."""
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = np.cov(Xc, rowvar=False)
        cov = np.atleast_2d(cov)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        self.explained_variance_ = eigvals[order]
        total = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total
        self.components_ = eigvecs[:, order].T
        return self

    def transform(self, X):
        """Project X onto the first n_components axes."""
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_[: self.n_components].T

    def inverse_transform(self, Z):
        """Reconstruct X from its low-dimensional projection Z."""
        Z = np.asarray(Z, dtype=float)
        return Z @ self.components_[: self.n_components] + self.mean_


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = rng.uniform(-5, 5, 200)
    X = np.column_stack([t, 0.2 * t + rng.normal(0, 0.3, 200)])
    pca = PCA(n_components=1).fit(X)
    print("explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))
    recon = pca.inverse_transform(pca.transform(X))
    print("reconstruction MSE:", round(float(np.mean((X - recon) ** 2)), 3))
