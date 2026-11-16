"""Elastic Net: least squares with a mix of L1 and L2 penalties

The elastic net (Zou and Hastie, 2005) mixes the two penalties
from the last two days: the L1 penalty of the lasso (Day 25),
which can set a coefficient to exactly zero, and the L2 penalty
of ridge (Day 24), which shrinks coefficients smoothly and
shares an effect across correlated features instead of picking
one of them. The mix is set by l1_ratio: 1.0 is the lasso, 0.0
is ridge, and values in between get some of both behaviors. In
particular, a group of near-copy features splits its effect
roughly evenly, where the lasso hands it to one of them. There
is no closed form, so this file fits by coordinate descent on
centered data, with the intercept left unpenalized: the update
for one coordinate is a soft-threshold (the L1 part) divided by
the squared column size plus the L2 strength (numpy only).
Run: python algorithms/elastic-net/impl.py
"""

import numpy as np


def _soft_threshold(rho, alpha):
    """Soft-thresholding operator: shrink rho toward zero by
    alpha, and return exactly zero when rho is within alpha
    of zero."""
    if rho > alpha:
        return rho - alpha
    if rho < -alpha:
        return rho + alpha
    return 0.0


def make_grouped(seed=0, n=300):
    """A group of three near-copy features that share an effect.

    The first three features are one latent factor plus tiny
    independent noise (spread 0.05), and the truth splits the
    group effect evenly between them: weights (1.5, 1.5, 1.5).
    A fourth feature works alone with weight 2.0, the last four
    are pure noise, and the intercept is 1.0 with noise of
    spread 0.5. Returns X, y, and the true weight vector.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, n)
    group = np.column_stack([
        z + rng.normal(0.0, 0.05, n) for _ in range(3)
    ])
    solo = rng.normal(0.0, 1.0, (n, 1))
    noise = rng.normal(0.0, 1.0, (n, 4))
    X = np.hstack([group, solo, noise])
    w_true = np.array([1.5, 1.5, 1.5, 2.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ w_true + 1.0 + rng.normal(0.0, 0.5, n)
    return X, y, w_true


def make_tiny_units(seed=0, n=300):
    """One real signal recorded in units a hundred times smaller.

    The signal is 3.0 times a standard normal factor, but the
    feature is stored divided by 100, as happens when a column
    is recorded in centimeters instead of meters. The true
    coefficient on the stored feature is therefore 300.0. The
    other three features are pure noise at the usual scale.
    Returns X, y, and the true weight vector.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, n)
    X = np.column_stack([
        z / 100.0,
        rng.normal(0.0, 1.0, (n, 3)),
    ])
    w_true = np.array([300.0, 0.0, 0.0, 0.0])
    y = 3.0 * z + rng.normal(0.0, 0.5, n)
    return X, y, w_true


class ElasticNet:
    """Elastic net regression fitted by coordinate descent,
    with an unpenalized intercept.

    Minimizes (1 / 2n) * ||y - X w - b||^2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2.
    fit centers X and y, sweeps the coordinates on the centered
    data, and recovers the intercept from the means. Features
    are used exactly as given, with no rescaling, so the penalty
    treats a coefficient of 300 the same whatever units the
    feature was recorded in. That choice is the point of the
    break-it demo below.
    """

    def __init__(self, alpha=0.1, l1_ratio=0.5, max_iter=1000,
                 tol=1e-6):
        if alpha < 0:
            raise ValueError("alpha must not be negative")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be between 0 and 1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if tol <= 0:
            raise ValueError("tol must be positive")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0

    def fit(self, X, y):
        """Train on (X, y). Returns self."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same length")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty")
        n, d = X.shape
        x_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - x_mean
        yc = y - y_mean
        col_sq = np.sum(Xc ** 2, axis=0) / n
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)
        w = np.zeros(d)
        for it in range(1, self.max_iter + 1):
            w_old = w.copy()
            resid = yc - Xc @ w
            for j in range(d):
                if col_sq[j] == 0.0:
                    w[j] = 0.0
                    continue
                # Partial residual correlation for coordinate j:
                # add its current effect back, then solve for it.
                # The L1 part soft-thresholds, the L2 part adds
                # to the denominator and shrinks smoothly.
                rho = (Xc[:, j] @ (resid + Xc[:, j] * w[j])) / n
                new = _soft_threshold(rho, l1) / (col_sq[j] + l2)
                if new != w[j]:
                    resid = resid + Xc[:, j] * (w[j] - new)
                    w[j] = new
            self.n_iter_ = it
            if np.max(np.abs(w - w_old)) < self.tol:
                break
        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w)
        return self

    def predict(self, X):
        """Predicted targets for each row of X."""
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """R squared of the predictions on (X, y)."""
        y = np.asarray(y, dtype=float)
        pred = self.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot

    def mse(self, X, y):
        """Mean squared error of the predictions on (X, y)."""
        y = np.asarray(y, dtype=float)
        return float(np.mean((y - self.predict(X)) ** 2))


if __name__ == "__main__":
    print("Elastic Net (seed 0)")

    # The works case: a correlated group shares its effect.
    X, y, w_true = make_grouped(seed=0)
    X_test, y_test, _ = make_grouped(seed=99)
    model = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
    print(f"grouped data: coef {np.round(model.coef_, 3)} "
          f"(true {w_true}), intercept {model.intercept_:.3f}, "
          f"held-out R2 {model.score(X_test, y_test):.3f}")
    print("The three near copies split the effect almost evenly, "
          "and the four noise features are exactly zero.")

    # What the lasso alone does with the same group.
    lasso = ElasticNet(alpha=0.1, l1_ratio=1.0).fit(X, y)
    print(f"same data, l1_ratio 1.0 (lasso): group coefficients "
          f"{np.round(lasso.coef_[:3], 3)}, spread "
          f"{np.max(lasso.coef_[:3]) - np.min(lasso.coef_[:3]):.3f} "
          f"against the elastic net spread "
          f"{np.max(model.coef_[:3]) - np.min(model.coef_[:3]):.3f}")

    # The break case: the same signal in units 100 times smaller.
    X, y, w_true = make_tiny_units(seed=0)
    X_test, y_test, _ = make_tiny_units(seed=99)
    tiny = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
    print(f"tiny units: coef {np.round(tiny.coef_, 3)} "
          f"(true {w_true}), held-out R2 "
          f"{tiny.score(X_test, y_test):.3f}")
    print("The only real feature is set to exactly zero, not "
          "because it carries nothing, but because in these "
          "units it would need a coefficient of 300 and the "
          "penalty prices coefficients, not effects. Record the "
          "same column at the usual scale and the fit recovers "
          "a coefficient near 3 with R2 above 0.95.")
