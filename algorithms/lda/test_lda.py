"""Tests for LDA: it separates classes whose means differ, and it
fails on classes that share a mean and differ only in spread."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_lda", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
LDA = _impl.LDA
make_blobs = _impl.make_blobs
make_same_mean = _impl.make_same_mean


def test_works():
    """On separated blobs, LDA gets high accuracy, its top
    eigenvalue is large, and the projected class centroids are
    far apart compared with the spread inside each class."""
    X, y = make_blobs(seed=0)
    model = LDA(n_components=2).fit(X, y)

    acc = model.score(X, y)
    assert acc > 0.95

    assert model.eigenvalues_[0] > 5.0
    assert model.components_.shape == (2, X.shape[1])

    Z = model.transform(X)
    centroids = model.centroids_
    between = float(np.linalg.norm(centroids[0] - centroids[1]))
    within = float(np.mean(np.linalg.norm(Z[y == 0] - centroids[0], axis=1)))
    assert between > 3.0 * within


def test_breaks():
    """Two classes with the same mean and different spreads break
    LDA: the between-class scatter is near zero, the top eigenvalue
    collapses, and accuracy stays near chance. A simple radius rule
    separates the same data well, so the failure is the mean-only
    view, not the data."""
    X, y = make_same_mean(seed=0)
    model = LDA(n_components=1).fit(X, y)

    acc = model.score(X, y)
    assert acc < 0.65
    assert model.eigenvalues_[0] < 0.5

    radius = np.linalg.norm(X, axis=1)
    by_radius = (radius > 1.2).astype(int)
    radius_acc = float(np.mean(by_radius == y))
    assert radius_acc > 0.75
    assert radius_acc > acc + 0.15
