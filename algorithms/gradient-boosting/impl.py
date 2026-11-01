"""Gradient boosting for regression, built from decision stumps.

Each round fits one depth-1 tree (a single split) to the current
residuals, then adds a shrunken copy of that tree to the model. The
first prediction is just the mean of y, and every round after that
corrects what is left over. Numpy only.
Run: python algorithms/gradient-boosting/impl.py
"""

import numpy as np


class _Stump:
    """One split: feature, threshold, and a value on each side."""

    def __init__(self, feature, threshold, left_value, right_value):
        self.feature = feature
        self.threshold = threshold
        self.left_value = left_value
        self.right_value = right_value

    def predict(self, X):
        left = X[:, self.feature] <= self.threshold
        out = np.where(left, self.left_value, self.right_value)
        return out


def _best_split(X, residual):
    """Best single split for squared loss, found exactly.

    For each feature, sort the points, then scan candidate thresholds
    between distinct values, tracking the sum of squared errors on
    each side with running sums. Returns (feature, threshold,
    left_value, right_value) or None when no split is possible.
    """
    n, n_features = X.shape
    total_sum = float(np.sum(residual))
    total_sq = float(np.sum(residual**2))
    best = None
    best_sse = total_sq
    for feature in range(n_features):
        order = np.argsort(X[:, feature], kind="stable")
        xs = X[order, feature]
        rs = residual[order]
        left_sum = 0.0
        left_sq = 0.0
        for i in range(n - 1):
            left_sum += float(rs[i])
            left_sq += float(rs[i] ** 2)
            if xs[i] == xs[i + 1]:
                continue
            left_n = i + 1
            right_n = n - left_n
            right_sum = total_sum - left_sum
            right_sq = total_sq - left_sq
            sse = (
                left_sq
                - left_sum**2 / left_n
                + right_sq
                - right_sum**2 / right_n
            )
            if sse < best_sse - 1e-12:
                best_sse = sse
                threshold = float((xs[i] + xs[i + 1]) / 2.0)
                best = (
                    feature,
                    threshold,
                    left_sum / left_n,
                    right_sum / right_n,
                )
    return best


class GradientBoostingRegressor:
    """Stagewise additive regression from decision stumps.

    Start at the mean of y. Each round, fit a stump to the residuals
    (y minus the current prediction) and add learning_rate times that
    stump to the model. Small learning rates with many rounds usually
    beat one big jump, but too many rounds will memorize noise, so
    n_estimators is a real tuning knob, not a free lunch.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1):
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Fit the stump sequence. Sets init_, stumps_, train_mse_."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(y) != len(X):
            raise ValueError("X and y must have the same length")
        self.init_ = float(np.mean(y))
        prediction = np.full(len(y), self.init_)
        self.stumps_ = []
        self.train_mse_ = []
        for _ in range(self.n_estimators):
            residual = y - prediction
            split = _best_split(X, residual)
            if split is None:
                break
            feature, threshold, left_value, right_value = split
            stump = _Stump(feature, threshold, left_value, right_value)
            self.stumps_.append(stump)
            prediction = prediction + self.learning_rate * stump.predict(X)
            self.train_mse_.append(float(np.mean((y - prediction) ** 2)))
        return self

    def predict(self, X):
        """Predict for new rows: mean of y plus every shrunken stump."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        prediction = np.full(len(X), self.init_)
        for stump in self.stumps_:
            prediction = prediction + self.learning_rate * stump.predict(X)
        return prediction


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 3.0, (200, 1))
    y = np.sin(2.0 * X[:, 0]) + rng.normal(0, 0.1, 200)
    X_train, y_train = X[:150], y[:150]
    X_test, y_test = X[150:], y[150:]
    baseline = float(np.mean((y_test - np.mean(y_train)) ** 2))
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1
    ).fit(X_train, y_train)
    test_mse = float(np.mean((y_test - model.predict(X_test)) ** 2))
    print("stumps fitted:", len(model.stumps_))
    print("baseline MSE (predict the mean):", round(baseline, 4))
    print("train MSE:", round(model.train_mse_[-1], 4))
    print("test MSE:", round(test_mse, 4))
    print("note: stumps add up to a curve, but each one is just a step")
