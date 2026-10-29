"""Tests for the decision tree: it learns blobs, and it memorizes noise."""

import os
import sys

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_decision_tree", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
DecisionTree, accuracy = _impl.DecisionTree, _impl.accuracy


def _blobs(rng, n_per_class, noise_frac=0.0):
    X = np.vstack(
        [rng.normal(0, 1, (n_per_class, 2)), rng.normal(0, 1, (n_per_class, 2)) + [4, 0]]
    )
    y = np.array([0] * n_per_class + [1] * n_per_class)
    if noise_frac:
        flip = rng.random(len(y)) < noise_frac
        y = np.where(flip, 1 - y, y)
    return X, y


def test_works():
    """A depth-limited tree learns clean blobs and generalizes."""
    rng = np.random.default_rng(0)
    X_train, y_train = _blobs(rng, 100)
    X_test, y_test = _blobs(rng, 50)
    tree = DecisionTree(max_depth=4).fit(X_train, y_train)
    assert accuracy(y_train, tree.predict(X_train)) > 0.95
    assert accuracy(y_test, tree.predict(X_test)) > 0.90


def test_breaks():
    """With no depth limit and 20 percent label noise, the tree memorizes:
    near-perfect train accuracy, a big gap to test accuracy."""
    rng = np.random.default_rng(0)
    X_train, y_train = _blobs(rng, 200, noise_frac=0.20)
    X_test, y_test = _blobs(rng, 100, noise_frac=0.20)
    tree = DecisionTree(max_depth=None).fit(X_train, y_train)
    train_acc = accuracy(y_train, tree.predict(X_train))
    test_acc = accuracy(y_test, tree.predict(X_test))
    assert train_acc > 0.99
    assert train_acc - test_acc > 0.10
