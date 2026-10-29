"""Tests for linear regression: it works, and one outlier bends it."""

import os
import sys

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_linear_regression", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
fit, mse, predict = _impl.fit, _impl.mse, _impl.predict


def test_works():
    """On clean data the closed form recovers the true line exactly."""
    X = np.linspace(0, 10, 50)
    y = 3.0 * X - 2.0
    coef = fit(X, y)
    assert abs(coef[0] + 2.0) < 1e-8
    assert abs(coef[1] - 3.0) < 1e-8
    assert mse(y, predict(X, coef)) < 1e-12


def test_breaks():
    """One huge outlier at the edge of the range drags the slope hard."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, 100)
    y = 2.0 * X + 1.0 + rng.normal(0, 1.0, 100)
    clean_slope = fit(X, y)[1]
    X_bad = np.append(X, 10.0)
    y_bad = np.append(y, 1000.0)
    bad_slope = fit(X_bad, y_bad)[1]
    assert abs(bad_slope - clean_slope) > 1.0
