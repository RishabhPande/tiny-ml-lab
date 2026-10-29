"""Linear Regression
Closed-form least squares linear regression (numpy only).
Run: python algorithms/linear-regression/impl.py
"""

import numpy as np


def fit(X, y):
    """Fit y ~= intercept + X @ slopes. Returns coef [intercept, slopes...]."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xb = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
    return coef


def predict(X, coef):
    """Predict with a coefficient vector from fit."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xb = np.column_stack([np.ones(len(X)), X])
    return Xb @ coef


def mse(y_true, y_pred):
    """Mean squared error."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, 100)
    y = 2.0 * X + 1.0 + rng.normal(0, 1.0, 100)
    coef = fit(X, y)
    print("true: intercept 1.0, slope 2.0")
    print("fitted coef:", np.round(coef, 3))
    print("train MSE:", round(mse(y, predict(X, coef)), 3))
