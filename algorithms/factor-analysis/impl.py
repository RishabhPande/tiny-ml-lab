"""Factor Analysis: explain many measured variables with a few
hidden factors, plus noise that belongs to each variable alone.

The model says each observation is a shared part plus a private
part: x = W z + noise, where z holds a small number of latent
factors, W (the loadings) says how strongly each factor shows
up in each measured variable, and the noise is diagonal, one
variance per variable, called the unique variance. Fitting is
done by expectation-maximisation. The E step works out what
the factors probably were for each point given the current
loadings, and the M step refits the loadings and the unique
variances from those guesses. Unlike PCA, the private noise is
part of the model, so a variable that is mostly noise does not
drag a factor toward itself. The catch is the factor count:
ask for one factor when the data really has two independent
ones and the single factor has to split its attention, leaving
about half of the shared signal unexplained (numpy only).
Run: python algorithms/factor-analysis/impl.py
"""

import numpy as np


def make_factor_data(seed=0, n=400, noise_std=0.3):
    """Two hidden factors showing up in six measured variables.

    Latent factors are standard normal, shape (n, 2). The true
    loadings, shape (6, 2), put factor 0 mainly on the first
    three variables and factor 1 mainly on the last three, with
    smaller cross loadings. Observations are factors times
    loadings transposed plus per-variable Gaussian noise with
    the given scale. Returns (X, Z, W_true).
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    W_true = np.array(
        [
            [1.2, 0.2],
            [1.0, -0.1],
            [0.9, 0.3],
            [0.2, 1.1],
            [-0.1, 1.0],
            [0.3, 0.8],
        ]
    )
    X = Z @ W_true.T + rng.normal(scale=noise_std, size=(n, W_true.shape[0]))
    return X, Z, W_true


def reconstruction_r2(X, X_reconstructed):
    """Fraction of total variance kept by a reconstruction."""
    X = np.asarray(X, dtype=float)
    X_reconstructed = np.asarray(X_reconstructed, dtype=float)
    ss_res = float(np.sum((X - X_reconstructed) ** 2))
    ss_tot = float(np.sum((X - X.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot


class FactorAnalysis:
    """Factor analysis fitted by expectation-maximisation.

    The loadings start from the top eigenvectors of the sample
    covariance, scaled by the square roots of their eigenvalues,
    and the unique variances start from what that first guess
    leaves unexplained, floored at 0.05 so no variable starts
    with zero private noise. Each EM round computes the
    posterior mean and covariance of the factors for every
    point, then updates the loadings by least squares against
    those posteriors and the unique variances from the residual
    diagonal. Iteration stops when the loadings barely move.
    """

    def __init__(self, n_factors=2, max_iter=500, tol=1e-7):
        if n_factors < 1:
            raise ValueError("n_factors must be at least 1")
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        n, d = X.shape
        if self.n_factors > d:
            raise ValueError("n_factors cannot exceed the number of variables")
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / n

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        k = self.n_factors
        W = eigvecs[:, :k] * np.sqrt(eigvals[:k])
        psi = np.maximum(np.diag(cov) - np.sum(W ** 2, axis=1), 0.05)

        for _ in range(self.max_iter):
            # E step: posterior of z given x is Gaussian with
            # covariance (I + W^T diag(1/psi) W)^-1 and mean
            # cov_zx = beta @ x, beta = cov_z @ W^T / psi.
            W_over_psi = W / psi[:, None]
            cov_z = np.linalg.inv(np.eye(k) + W.T @ W_over_psi)
            beta = cov_z @ W_over_psi.T
            Ez = Xc @ beta.T
            Ezz_sum = n * cov_z + Ez.T @ Ez

            # M step: loadings by least squares, then the
            # unique variances from the leftover diagonal.
            W_new = (Xc.T @ Ez) @ np.linalg.inv(Ezz_sum)
            psi_new = np.maximum(np.diag(cov - W_new @ beta @ cov), 1e-6)
            change = float(np.max(np.abs(W_new - W)))
            W, psi = W_new, psi_new
            if change < self.tol:
                break

        self.loadings_ = W
        self.noise_variance_ = psi
        return self

    def transform(self, X):
        """Posterior mean of the factors for each row of X."""
        X = np.asarray(X, dtype=float)
        Xc = X - self.mean_
        W = self.loadings_
        psi = self.noise_variance_
        k = self.n_factors
        W_over_psi = W / psi[:, None]
        cov_z = np.linalg.inv(np.eye(k) + W.T @ W_over_psi)
        beta = cov_z @ W_over_psi.T
        return Xc @ beta.T

    def reconstruct(self, X):
        """Rebuild X from its factor scores, plus the mean."""
        return self.transform(X) @ self.loadings_.T + self.mean_


if __name__ == "__main__":
    X, Z, W_true = make_factor_data(seed=0)

    print("Factor analysis with 2 factors on 6 variables (seed 0)")
    model = FactorAnalysis(n_factors=2)
    model.fit(X)
    rebuilt = model.reconstruct(X)
    print(f"reconstruction R2: {reconstruction_r2(X, rebuilt):.3f}")
    print(f"unique variances: {np.round(model.noise_variance_, 3)}")
    print("the true per-variable noise variance is 0.09, and the")
    print("estimates land near it, so the private noise stays")
    print("private instead of leaking into the factors")

    print("\nFactor analysis with 1 factor on the same data (seed 0)")
    one = FactorAnalysis(n_factors=1)
    one.fit(X)
    rebuilt_one = one.reconstruct(X)
    print(f"reconstruction R2: {reconstruction_r2(X, rebuilt_one):.3f}")
    print("one factor cannot be two independent factors at once,")
    print("so about half of the shared signal is left unexplained")
