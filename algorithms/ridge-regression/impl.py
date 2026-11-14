"""Ridge Regression: least squares with an L2 penalty

Ridge regression (Hoerl and Kennard, 1970) fits a linear model
the same way ordinary least squares does, then adds a penalty
on the size of the coefficients. Solving with the penalty means
solving (X^T X + alpha I) w = X^T y on centered data, with the
intercept left unpenalized. A small alpha changes almost
nothing. A moderate alpha matters most when features are
near copies of each other: least squares can assign huge,
opposing coefficients that cancel on the training data and
wobble on new data, while ridge keeps the coefficients small
and shared across the copies, at almost no cost in accuracy.
Push alpha far enough, though, and the penalty wins outright:
the coefficients shrink toward zero, the model says almost
nothing, and the fit collapses (numpy only).
Run: python algorithms/ridge-regression/impl.py
"""

import numpy as np


def make_regression(seed=0, n=200):
    """Clean linear data with known coefficients.

    Three independent Gaussian features, true weights
    (3.0, -2.0, 1.5), intercept 2.0, and noise with spread 0.5.
    Returns X, y, and the true weight vector.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, 3))
    w_true = np.array([3.0, -2.0, 1.5])
    y = X @ w_true + 2.0 + rng.normal(0.0, 0.5, n)
    return X, y, w_true


def make_collinear(seed=0, n=120, noise=0.5):
    """Three features that are near copies of one hidden factor.

    Each feature is the same latent z plus tiny independent
    noise (spread 0.05), so the features carry almost the same
    information. True weights are (2.0, 1.0, -1.0), intercept
    0.0. Returns X, y, and the true weight vector.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, (n, 1))
    X = np.hstack([z + rng.normal(0.0, 0.05, (n, 1)) for _ in range(3)])
    w_true = np.array([2.0, 1.0, -1.0])
    y = (X @ w_true).ravel() + rng.normal(0.0, noise, n)
    return X, y, w_true


class Ridge:
    """Ridge regression with an unpenalized intercept.

    fit centers X and y, solves (X^T X + alpha I) w = X^T y on
    the centered data, and recovers the intercept from the
    means. alpha = 0 is ordinary least squares.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        """Train on (X, y). Returns self."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        x_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - x_mean
        yc = y - y_mean
        system = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        try:
            self.coef_ = np.linalg.solve(system, Xc.T @ yc)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.pinv(system) @ (Xc.T @ yc)
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
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
    print("Ridge Regression (seed 0)")

    # The works case: clean data, small penalty, true weights back.
    X, y, w_true = make_regression(seed=0)
    X_test, y_test, _ = make_regression(seed=99)
    model = Ridge(alpha=1.0).fit(X, y)
    print(f"clean data: coef {np.round(model.coef_, 3)} "
          f"(true {w_true}), intercept {model.intercept_:.3f}, "
          f"held-out R2 {model.score(X_test, y_test):.3f}")

    # Why the penalty exists: near-duplicate features.
    X, y, _ = make_collinear(seed=0)
    X_test, y_test, _ = make_collinear(seed=99)
    ols = Ridge(alpha=0.0).fit(X, y)
    ridge = Ridge(alpha=10.0).fit(X, y)
    print(f"collinear data: least squares coef norm "
          f"{np.linalg.norm(ols.coef_):.3f}, held-out R2 "
          f"{ols.score(X_test, y_test):.3f}")
    print(f"collinear data: ridge alpha 10 coef norm "
          f"{np.linalg.norm(ridge.coef_):.3f}, held-out R2 "
          f"{ridge.score(X_test, y_test):.3f}")

    # The break case: the penalty turned up until nothing is left.
    X, y, _ = make_regression(seed=0)
    X_test, y_test, _ = make_regression(seed=99)
    crushed = Ridge(alpha=1e6).fit(X, y)
    print(f"alpha 1e6: coef {np.round(crushed.coef_, 4)}, "
          f"held-out R2 {crushed.score(X_test, y_test):.3f}")
    print("Too much penalty does not play it safe. It erases "
          "the signal along with the noise.")
