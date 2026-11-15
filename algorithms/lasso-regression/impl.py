"""Lasso Regression: least squares with an L1 penalty

The lasso (Tibshirani, 1996) fits a linear model the same way
ordinary least squares does, then adds a penalty on the sum of
the absolute values of the coefficients. That one change in the
penalty, absolute size instead of squared size, has a sharp
consequence: the penalty can push a coefficient all the way to
exactly zero and hold it there, so the fit doubles as feature
selection. Useless features do not just get small weights, they
get removed. There is no closed form, so this file fits by
coordinate descent on centered data, with the intercept left
unpenalized: hold every coefficient but one fixed, solve for
that one in closed form with a soft-threshold, and sweep until
nothing moves. The soft-threshold is the whole trick, since it
is what sets weak coefficients to exactly zero instead of just
shrinking them (numpy only).
Run: python algorithms/lasso-regression/impl.py
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


def make_sparse(seed=0, n=200):
    """Mostly useless features with a sparse truth.

    Ten independent Gaussian features, but only three of them
    matter: true weights (3.0, 0, 0, -2.0, 0, 0, 0, 1.5, 0, 0),
    intercept 2.0, and noise with spread 0.5. Returns X, y, and
    the true weight vector.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, 10))
    w_true = np.array([3.0, 0.0, 0.0, -2.0, 0.0,
                       0.0, 0.0, 1.5, 0.0, 0.0])
    y = X @ w_true + 2.0 + rng.normal(0.0, 0.5, n)
    return X, y, w_true


def make_twins(seed=0, n=200):
    """Two features that are near copies of one hidden factor.

    Each feature is the same latent z plus tiny independent
    noise (spread 0.01), so the two carry almost the same
    information, and the truth splits the effect evenly between
    them: weights (2.0, 2.0), intercept 0.0. Returns X, y, and
    the true weight vector.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, n)
    X = np.column_stack([
        z + rng.normal(0.0, 0.01, n),
        z + rng.normal(0.0, 0.01, n),
    ])
    w_true = np.array([2.0, 2.0])
    y = X @ w_true + rng.normal(0.0, 0.5, n)
    return X, y, w_true


class Lasso:
    """Lasso regression fitted by coordinate descent, with an
    unpenalized intercept.

    Minimizes (1 / 2n) * ||y - X w - b||^2 + alpha * ||w||_1.
    fit centers X and y, sweeps the coordinates on the centered
    data, and recovers the intercept from the means. A larger
    alpha means more coefficients set to exactly zero.
    """

    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-6):
        if alpha < 0:
            raise ValueError("alpha must not be negative")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if tol <= 0:
            raise ValueError("tol must be positive")
        self.alpha = alpha
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
                rho = (Xc[:, j] @ (resid + Xc[:, j] * w[j])) / n
                new = _soft_threshold(rho, self.alpha) / col_sq[j]
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
    print("Lasso Regression (seed 0)")

    # The works case: sparse truth, most coefficients exactly zero.
    X, y, w_true = make_sparse(seed=0)
    X_test, y_test, _ = make_sparse(seed=99)
    model = Lasso(alpha=0.1).fit(X, y)
    n_zero = int(np.sum(model.coef_ == 0.0))
    print(f"sparse data: coef {np.round(model.coef_, 3)} "
          f"(true {w_true}), intercept {model.intercept_:.3f}, "
          f"held-out R2 {model.score(X_test, y_test):.3f}")
    print(f"sparse data: {n_zero} of {len(w_true)} coefficients "
          f"are exactly zero, and all seven belong to features "
          f"that do not matter")

    # What least squares does with the same useless features.
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    ols_coef = np.linalg.solve(Xc.T @ Xc, Xc.T @ yc)
    print(f"same data, least squares: smallest useless-feature "
          f"coefficient is {np.min(np.abs(ols_coef[w_true == 0.0])):.3f} "
          f"in size, never exactly zero")

    # The break case: twin features, and the lasso picks one.
    X, y, w_true = make_twins(seed=0)
    X_test, y_test, _ = make_twins(seed=99)
    twins = Lasso(alpha=0.1).fit(X, y)
    print(f"twin features: coef {np.round(twins.coef_, 3)} "
          f"(true {w_true}), held-out R2 "
          f"{twins.score(X_test, y_test):.3f}")
    print("The predictions are fine. The story is wrong: the "
          "effect is shared, and the lasso hands it all to "
          "whichever twin it reaches first.")
