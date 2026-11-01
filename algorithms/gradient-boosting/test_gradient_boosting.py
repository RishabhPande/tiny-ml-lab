"""Tests for gradient boosting: stumps add up to a curve, but a tree
model cannot extrapolate past the range it trained on."""

import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "impl_gradient_boosting",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
GradientBoostingRegressor = _impl.GradientBoostingRegressor


def _sine_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 3.0, (200, 1))
    y = np.sin(2.0 * X[:, 0]) + rng.normal(0, 0.1, 200)
    return X, y


def test_works():
    """Boosted stumps learn a sine curve far better than the mean.

    The mean predictor scores around 0.5 MSE on this curve. One
    hundred stumps at learning rate 0.1 should land well under 0.05
    on held-out points drawn from the same range, because each round
    corrects the leftover error of the rounds before it.
    """
    X, y = _sine_data()
    X_train, y_train = X[:150], y[:150]
    X_test, y_test = X[150:], y[150:]
    baseline = float(np.mean((y_test - np.mean(y_train)) ** 2))
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1
    ).fit(X_train, y_train)
    test_mse = float(np.mean((y_test - model.predict(X_test)) ** 2))
    assert test_mse < 0.05
    assert test_mse < baseline / 5.0


def test_breaks():
    """A boosted tree model cannot extrapolate beyond its training range.

    Train on a straight line y = x with x between 0 and 1, then ask
    for x = 5. The true value is 5, but every stump is a step that
    returns its right-side value for anything past its threshold, so
    the prediction stays stuck near the largest training value. Any
    tree ensemble, boosted or not, is flat outside the data it saw.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (100, 1))
    y = X[:, 0]
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1
    ).fit(X, y)
    prediction = float(model.predict(np.array([[5.0]]))[0])
    assert prediction < 2.0
    assert abs(prediction - 5.0) > 3.0


def test_more_rounds_lower_train_error():
    """Training error falls as rounds are added, which is the point.

    Each stump is fit to the current residuals, so the training MSE
    after 50 rounds must be lower than after 5 rounds on the same
    data. This is also the warning: the same mechanism that drives
    training error down will eventually memorize noise.
    """
    X, y = _sine_data()
    few = GradientBoostingRegressor(
        n_estimators=5, learning_rate=0.1
    ).fit(X, y)
    many = GradientBoostingRegressor(
        n_estimators=50, learning_rate=0.1
    ).fit(X, y)
    assert many.train_mse_[-1] < few.train_mse_[-1]


def test_same_data_same_predictions():
    """Fitting twice on the same data gives identical predictions."""
    X, y = _sine_data()
    a = GradientBoostingRegressor(
        n_estimators=20, learning_rate=0.1
    ).fit(X, y)
    b = GradientBoostingRegressor(
        n_estimators=20, learning_rate=0.1
    ).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))
