"""Tests for kNN: it aces clean blobs, and noisy dimensions sink it."""

import os
import sys

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_knn", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
KNNClassifier, accuracy = _impl.KNNClassifier, _impl.accuracy


def _blobs(rng, n_per_class, n_noise=0, noise_scale=3.0):
    """Two blobs separated on the first two features, plus noise features."""
    base_train = np.vstack(
        [rng.normal(0, 1, (n_per_class, 2)), rng.normal(0, 1, (n_per_class, 2)) + [4, 0]]
    )
    base_test = np.vstack(
        [rng.normal(0, 1, (n_per_class // 2, 2)), rng.normal(0, 1, (n_per_class // 2, 2)) + [4, 0]]
    )
    y_train = np.array([0] * n_per_class + [1] * n_per_class)
    y_test = np.array([0] * (n_per_class // 2) + [1] * (n_per_class // 2))
    if n_noise:
        noise_train = rng.uniform(-noise_scale, noise_scale, (2 * n_per_class, n_noise))
        noise_test = rng.uniform(-noise_scale, noise_scale, (n_per_class, n_noise))
        return (
            np.hstack([base_train, noise_train]),
            y_train,
            np.hstack([base_test, noise_test]),
            y_test,
        )
    return base_train, y_train, base_test, y_test


def test_works():
    """On clean 2D blobs, kNN is nearly perfect."""
    rng = np.random.default_rng(0)
    X_train, y_train, X_test, y_test = _blobs(rng, 200)
    knn = KNNClassifier(n_neighbors=5).fit(X_train, y_train)
    assert accuracy(y_test, knn.predict(X_test)) > 0.95


def test_breaks():
    """With 50 noisy features, distance stops meaning anything."""
    rng = np.random.default_rng(0)
    X_train, y_train, X_test, y_test = _blobs(rng, 200)
    clean = KNNClassifier(n_neighbors=5).fit(X_train, y_train)
    clean_acc = accuracy(y_test, clean.predict(X_test))
    X_train_n, y_train_n, X_test_n, y_test_n = _blobs(rng, 200, n_noise=50)
    noisy = KNNClassifier(n_neighbors=5).fit(X_train_n, y_train_n)
    noisy_acc = accuracy(y_test_n, noisy.predict(X_test_n))
    assert noisy_acc < clean_acc - 0.15
