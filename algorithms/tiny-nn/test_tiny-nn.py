"""Tests for the tiny NN: it solves XOR, and unscaled features stall it."""

import os
import sys

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_tiny_nn", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
MLP, accuracy = _impl.MLP, _impl.accuracy


def test_works():
    """A hidden layer makes XOR learnable: accuracy reaches 1.0."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    mlp = MLP(hidden=8, lr=0.5, epochs=5000, seed=0).fit(X, y)
    assert accuracy(y, mlp.predict(X)) >= 0.95


def _scale_data(rng, n_per_class):
    """Classes separate on feature 1; feature 2 is uniform noise 0..1000."""
    x1_0 = rng.normal(0, 0.2, n_per_class)
    x1_1 = rng.normal(1, 0.2, n_per_class)
    x2 = rng.uniform(0, 1000, 2 * n_per_class)
    X_train = np.column_stack(
        [np.concatenate([x1_0, x1_1]), x2[: 2 * n_per_class]]
    )
    y_train = np.array([0] * n_per_class + [1] * n_per_class)
    x1_0t = rng.normal(0, 0.2, n_per_class // 2)
    x1_1t = rng.normal(1, 0.2, n_per_class // 2)
    x2t = rng.uniform(0, 1000, n_per_class)
    X_test = np.column_stack([np.concatenate([x1_0t, x1_1t]), x2t])
    y_test = np.array([0] * (n_per_class // 2) + [1] * (n_per_class // 2))
    return X_train, y_train, X_test, y_test


def test_breaks():
    """Without scaling, the 0..1000 noise feature saturates tanh and the
    network learns nothing; with scaling, the same network learns fine."""
    rng = np.random.default_rng(0)
    X_train, y_train, X_test, y_test = _scale_data(rng, 200)
    raw = MLP(hidden=8, lr=0.1, epochs=300, seed=0).fit(X_train, y_train)
    raw_acc = accuracy(y_test, raw.predict(X_test))
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    scaled = MLP(hidden=8, lr=0.1, epochs=300, seed=0).fit(
        (X_train - mu) / sd, y_train
    )
    scaled_acc = accuracy(y_test, scaled.predict((X_test - mu) / sd))
    assert scaled_acc > 0.9
    assert raw_acc < 0.8
