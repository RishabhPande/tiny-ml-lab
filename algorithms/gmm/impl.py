"""Gaussian Mixture Model: soft clustering by fitting a mix of
Gaussian blobs with expectation-maximization (numpy only).

K-means assigns each point to exactly one cluster. A mixture model
is softer: each point gets a responsibility for every component, the
probability that it came from that Gaussian. The E step computes
those responsibilities from the current means, covariances, and
weights. The M step refits each Gaussian to the points it is
responsible for, weighted by those responsibilities. Repeating the
two steps climbs the data log-likelihood until it stops improving.

Run: python algorithms/gmm/impl.py
"""

import numpy as np


def _log_gaussian(X, mean, cov):
    """Log density of a multivariate Gaussian at each row of X."""
    d = X.shape[1]
    cov = cov + 1e-9 * np.eye(d)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + 1e-6 * np.eye(d)
        sign, logdet = np.linalg.slogdet(cov)
    diff = X - mean
    solved = np.linalg.solve(cov, diff.T).T
    quad = np.sum(diff * solved, axis=1)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


class GaussianMixture:
    """Gaussian mixture fitted by expectation-maximization.

    Each component has its own full covariance matrix, so clusters
    can be stretched and rotated, not just round. Covariances get a
    small ridge (reg_covar times the identity) at every M step, which
    keeps a component from collapsing onto a single point and its
    covariance from going singular.
    """

    def __init__(self, n_components=2, max_iter=100, tol=1e-4,
                 reg_covar=1e-6, seed=42):
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if reg_covar < 0:
            raise ValueError("reg_covar must not be negative")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.seed = seed

    def fit(self, X):
        """Fit the mixture. Sets weights_, means_, covariances_,
        responsibilities_, log_likelihood_, log_likelihood_history_,
        labels_, n_iter_, and converged_."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] < self.n_components:
            raise ValueError(
                "X must have at least as many rows as n_components"
            )
        if X.shape[0] == 0:
            raise ValueError("X must not be empty")
        n, d = X.shape
        rng = np.random.default_rng(self.seed)

        # Init: means start on distinct data points, covariances start
        # at the overall data covariance, weights start uniform.
        starts = rng.choice(n, size=self.n_components, replace=False)
        means = X[starts].copy()
        global_cov = np.cov(X.T) + self.reg_covar * np.eye(d)
        if global_cov.ndim == 0:
            global_cov = np.atleast_2d(global_cov)
        covariances = np.array([global_cov.copy()
                                for _ in range(self.n_components)])
        weights = np.full(self.n_components, 1.0 / self.n_components)

        prev_ll = -np.inf
        history = []
        responsibilities = None
        for it in range(1, self.max_iter + 1):
            # E step: log responsibilities, normalized with logsumexp.
            log_prob = np.column_stack([
                np.log(weights[k]) + _log_gaussian(X, means[k],
                                                   covariances[k])
                for k in range(self.n_components)
            ])
            log_max = np.max(log_prob, axis=1, keepdims=True)
            log_sum = log_max + np.log(
                np.sum(np.exp(log_prob - log_max), axis=1, keepdims=True)
            )
            ll = float(np.sum(log_sum))
            history.append(ll)
            responsibilities = np.exp(log_prob - log_sum)

            # M step: refit each Gaussian to its responsibilities.
            counts = responsibilities.sum(axis=0) + 1e-12
            weights = counts / n
            means = (responsibilities.T @ X) / counts[:, None]
            for k in range(self.n_components):
                diff = X - means[k]
                weighted = responsibilities[:, k][:, None] * diff
                covariances[k] = (
                    weighted.T @ diff
                ) / counts[k] + self.reg_covar * np.eye(d)

            if abs(ll - prev_ll) < self.tol:
                self.n_iter_ = it
                self.converged_ = True
                break
            prev_ll = ll
        else:
            self.n_iter_ = self.max_iter
            self.converged_ = False

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.responsibilities_ = responsibilities
        self.log_likelihood_ = history[-1]
        self.log_likelihood_history_ = history
        self.labels_ = np.argmax(responsibilities, axis=1)
        return self

    def predict_proba(self, X):
        """Responsibility of each component for each row of X."""
        X = np.asarray(X, dtype=float)
        log_prob = np.column_stack([
            np.log(self.weights_[k]) + _log_gaussian(
                X, self.means_[k], self.covariances_[k]
            )
            for k in range(self.n_components)
        ])
        log_max = np.max(log_prob, axis=1, keepdims=True)
        probs = np.exp(log_prob - log_max)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Hard label: the component with the highest responsibility."""
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X):
        """Mean log-likelihood of X under the fitted mixture."""
        X = np.asarray(X, dtype=float)
        log_prob = np.column_stack([
            np.log(self.weights_[k]) + _log_gaussian(
                X, self.means_[k], self.covariances_[k]
            )
            for k in range(self.n_components)
        ])
        log_max = np.max(log_prob, axis=1, keepdims=True)
        log_sum = log_max + np.log(
            np.sum(np.exp(log_prob - log_max), axis=1, keepdims=True)
        )
        return float(np.mean(log_sum))


def _make_blobs(seed=42):
    """Two well separated Gaussian blobs, 150 points each."""
    rng = np.random.default_rng(seed)
    left = rng.normal(0, 1.0, (150, 2)) + np.array([-3.0, 0.0])
    right = rng.normal(0, 1.0, (150, 2)) + np.array([3.0, 0.0])
    X = np.vstack([left, right])
    y = np.array([0] * 150 + [1] * 150)
    return X, y


if __name__ == "__main__":
    X, y = _make_blobs()
    model = GaussianMixture().fit(X)
    print(f"converged: {model.converged_} in {model.n_iter_} iterations")
    print(f"log-likelihood: {model.log_likelihood_:.1f}")
    print(f"weights: {np.round(model.weights_, 3)}")
    print(f"means:\n{np.round(model.means_, 3)}")
    # Labels are up to permutation, so match each recovered mean to
    # its nearest true center before scoring accuracy.
    true_centers = np.array([[-3.0, 0.0], [3.0, 0.0]])
    order = [int(np.argmin(np.linalg.norm(
        true_centers - m, axis=1))) for m in model.means_]
    remapped = np.array([order[label] for label in model.labels_])
    print(f"clustering accuracy (up to label swap): "
          f"{np.mean(remapped == y):.3f}")
    print(f"soft assignments: mean max responsibility "
          f"{np.mean(np.max(model.responsibilities_, axis=1)):.3f}")
