"""Tests for logistic regression: it separates blobs, and it fails on XOR."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_logistic_regression", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
LogisticRegression = _impl.LogisticRegression


def test_works():
    """On two separated blobs, accuracy is above 0.95 and weights point right."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 1.0, (100, 2))
    X1 = rng.normal(0, 1.0, (100, 2)) + np.array([3.0, 3.0])
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)
    model = LogisticRegression(lr=0.5, n_iter=1000).fit(X, y)
    acc = float(np.mean(model.predict(X) == y))
    assert acc > 0.95
    assert model.weights_[1] > 0 and model.weights_[2] > 0
    probs = model.predict_proba(X)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_breaks():
    """On XOR, no straight line separates the classes, so accuracy stalls."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [0.0, 3.0], [3.0, 0.0], [3.0, 3.0]])
    labels = np.array([0, 1, 1, 0])
    X = np.vstack([c + rng.normal(0, 0.4, (50, 2)) for c in centers])
    y = np.repeat(labels, 50)
    model = LogisticRegression(lr=0.5, n_iter=2000).fit(X, y)
    acc = float(np.mean(model.predict(X) == y))
    assert acc < 0.75
