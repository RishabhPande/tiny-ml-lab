"""Tests for DBSCAN: it finds dense blobs and noise, and one radius
cannot handle clusters of very different densities."""

import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "impl_dbscan", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
DBSCAN = _impl.DBSCAN


def _two_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.4, (80, 2))
    b = rng.normal(0, 0.4, (80, 2)) + np.array([4.0, 4.0])
    return np.vstack([a, b])


def test_works():
    """Two separated blobs become exactly two clusters.

    The blobs are tight (spread 0.4) and far apart (centers 4 units
    apart in each axis), so eps 0.5 links points inside a blob and
    never links across blobs. At most a few stray tail points may
    land as noise, which is DBSCAN working as designed.
    """
    X = _two_blobs()
    model = DBSCAN(eps=0.5, min_samples=5).fit(X)
    labels = model.labels_
    clusters = sorted(set(labels.tolist()) - {-1})
    assert clusters == [0, 1]
    assert model.n_clusters_ == 2
    assert int(np.sum(labels == 0)) >= 70
    assert int(np.sum(labels == 1)) >= 70
    assert int(np.sum(labels == -1)) <= 5


def test_breaks():
    """One radius cannot serve a dense and a sparse cluster at once.

    The dense blob (spread 0.2) is found cleanly, but the sparse blob
    (spread 1.2) has too few points within any 0.5 radius, so most of
    its points are labeled noise even though they form a real group.
    Raising eps to rescue the sparse blob would risk merging things
    that should stay apart, which is the tradeoff DBSCAN forces.
    """
    rng = np.random.default_rng(2)
    dense = rng.normal(0, 0.2, (100, 2))
    sparse = rng.normal(0, 1.2, (100, 2)) + np.array([6.0, 0.0])
    X = np.vstack([dense, sparse])
    model = DBSCAN(eps=0.5, min_samples=10).fit(X)
    labels = model.labels_
    dense_noise = float(np.mean(labels[:100] == -1))
    sparse_noise = float(np.mean(labels[100:] == -1))
    assert dense_noise < 0.1
    assert sparse_noise > 0.5


def test_scattered_points_are_noise():
    """Isolated points with no close neighbors are labeled -1."""
    X = _two_blobs()
    far = np.array([[20.0, 20.0], [-20.0, -20.0], [20.0, -20.0]])
    X = np.vstack([X, far])
    labels = DBSCAN(eps=0.5, min_samples=5).fit(X).labels_
    assert list(labels[-3:]) == [-1, -1, -1]


def test_same_seed_same_labels():
    """Fitting twice on the same data gives identical labels."""
    X = _two_blobs()
    a = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
    b = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
    assert np.array_equal(a, b)
