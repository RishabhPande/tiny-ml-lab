"""Tests for the linear SVM: one that proves it works, one that breaks it."""

import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "impl_svm", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
SVMClassifier = _impl.SVMClassifier


def _two_blobs(seed=42):
    rng = np.random.default_rng(seed)
    neg = rng.normal(0, 0.8, (100, 2)) + np.array([-2.0, 0.0])
    pos = rng.normal(0, 0.8, (100, 2)) + np.array([2.0, 0.0])
    X = np.vstack([neg, pos])
    y = np.array([-1] * 100 + [1] * 100)
    return X, y


def _xor_blobs(seed=7):
    rng = np.random.default_rng(seed)
    corners = [
        (np.array([-2.0, -2.0]), 1),
        (np.array([2.0, 2.0]), 1),
        (np.array([-2.0, 2.0]), -1),
        (np.array([2.0, -2.0]), -1),
    ]
    xs, ys = [], []
    for center, label in corners:
        xs.append(rng.normal(0, 0.5, (50, 2)) + center)
        ys += [label] * 50
    return np.vstack(xs), np.array(ys)


def test_works():
    """On two separated blobs the SVM must find a boundary that
    classifies nearly everything correctly, with a nonzero weight
    vector pointing roughly along the axis that separates them."""
    X, y = _two_blobs()
    model = SVMClassifier().fit(X, y)
    assert model.accuracy(X, y) >= 0.97
    assert np.linalg.norm(model.coef_) > 0
    assert abs(model.coef_[0]) > abs(model.coef_[1])
    assert model.coef_[0] > 0
    assert model.n_steps_ == 500


def test_breaks():
    """XOR is not linearly separable: no straight line can put both
    diagonal corners on the right side. A linear SVM tops out near
    75 percent here, so it must score well below its blob accuracy,
    proving the boundary, not the optimizer, is the limit."""
    X, y = _xor_blobs()
    model = SVMClassifier().fit(X, y)
    assert model.accuracy(X, y) < 0.85


def test_rejects_bad_labels():
    """Labels outside -1, 0, and +1, and single-class data, are
    user errors, not training data, so fit must raise."""
    X, _ = _two_blobs()
    try:
        SVMClassifier().fit(X, np.full(200, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for label 2")
    try:
        SVMClassifier().fit(X, np.full(200, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for single-class y")
