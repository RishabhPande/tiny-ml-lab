"""Tests for random forest: it separates blobs, and majority vote erases
a rare, overlapping class."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_random_forest", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
RandomForestClassifier = _impl.RandomForestClassifier


def _blobs():
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 1.0, (100, 2))
    X1 = rng.normal(0, 1.0, (100, 2)) + np.array([3.0, 3.0])
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)
    return X, y


def test_works():
    """On held-out separated blobs, the forest beats 0.95 accuracy.

    Held-out on purpose: deep trees memorize their training rows, so a
    train accuracy here would prove nothing.
    """
    X, y = _blobs()
    perm = np.random.default_rng(42).permutation(len(y))
    train, test = perm[:140], perm[140:]
    model = RandomForestClassifier(n_estimators=50, seed=0).fit(X[train], y[train])
    acc = float(np.mean(model.predict(X[test]) == y[test]))
    assert acc > 0.95


def test_breaks():
    """A rare, overlapping class gets voted away on held-out data.

    The minority class is 5 percent of the data and sits close to the
    majority blob, so its leaves are rarely pure and the majority vote
    almost never picks it. Overall accuracy still looks fine, which is
    exactly why accuracy alone hides this failure: minority recall is 0.
    """
    rng = np.random.default_rng(1)
    X_major = rng.normal(0, 1.0, (760, 2))
    X_minor = rng.normal(0, 1.0, (40, 2)) + np.array([0.7, 0.7])
    X = np.vstack([X_major, X_minor])
    y = np.array([0] * 760 + [1] * 40)
    perm = np.random.default_rng(123).permutation(len(y))
    train, test = perm[:400], perm[400:]
    model = RandomForestClassifier(n_estimators=50, seed=0).fit(X[train], y[train])
    pred = model.predict(X[test])
    minority = y[test] == 1
    recall = float(np.mean(pred[minority] == 1))
    assert recall < 0.5


def test_proba_and_importances():
    """Probabilities sum to 1 per row and importances sum to 1 overall."""
    X, y = _blobs()
    model = RandomForestClassifier(n_estimators=20, seed=0).fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (200, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert np.isclose(model.feature_importances_.sum(), 1.0)


def test_same_seed_same_predictions():
    """Two forests with the same seed make identical predictions."""
    X, y = _blobs()
    a = RandomForestClassifier(n_estimators=20, seed=7).fit(X, y)
    b = RandomForestClassifier(n_estimators=20, seed=7).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))
