"""Tests for naive Bayes: independence works, duplication breaks it."""

import os
import sys

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_naive_bayes", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
GaussianNB, accuracy = _impl.GaussianNB, _impl.accuracy


def _train_data(rng, n_per_class=500):
    """Feature 1 separates the classes; feature 2 barely does."""
    f1_0 = rng.normal(0, 1, n_per_class)
    f1_1 = rng.normal(2, 1, n_per_class)
    f2_0 = rng.normal(0, 1, n_per_class)
    f2_1 = rng.normal(0.2, 1, n_per_class)
    X = np.column_stack(
        [np.concatenate([f1_0, f1_1]), np.concatenate([f2_0, f2_1])]
    )
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def test_works():
    """On independent features, naive Bayes is nearly perfect."""
    rng = np.random.default_rng(0)
    X_train = np.vstack(
        [rng.normal(0, 1, (200, 2)), rng.normal(0, 1, (200, 2)) + [3, 3]]
    )
    y_train = np.array([0] * 200 + [1] * 200)
    X_test = np.vstack(
        [rng.normal(0, 1, (100, 2)), rng.normal(0, 1, (100, 2)) + [3, 3]]
    )
    y_test = np.array([0] * 100 + [1] * 100)
    nb = GaussianNB().fit(X_train, y_train)
    assert accuracy(y_test, nb.predict(X_test)) > 0.95


def test_breaks():
    """Duplicating a weak feature 100 times lets it outvote a strong one."""
    rng = np.random.default_rng(0)
    X, y = _train_data(rng)
    # A point that is clearly class 1 on feature 1, class 0 on feature 2.
    point = np.array([[2.5, -1.5]])
    base = GaussianNB().fit(X, y)
    assert base.predict(point)[0] == 1
    X_dup = np.hstack([X[:, [0]], np.repeat(X[:, [1]], 100, axis=1)])
    point_dup = np.hstack([point[:, [0]], np.repeat(point[:, [1]], 100, axis=1)])
    dup = GaussianNB().fit(X_dup, y)
    assert dup.predict(point_dup)[0] == 0
