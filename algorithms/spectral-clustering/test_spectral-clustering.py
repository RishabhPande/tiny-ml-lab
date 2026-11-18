"""Tests for Spectral Clustering: the eigenvector embedding
separates two concentric rings exactly, a shape no center
based method can split, but it breaks when sigma is far too
small, because the similarity graph falls apart into near
isolated points and the eigenvectors stop describing the data.
"""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_spectral_clustering",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
SpectralClustering = _impl.SpectralClustering
make_rings = _impl.make_rings
make_blobs = _impl.make_blobs
best_accuracy = _impl._best_accuracy


def test_works():
    """On two concentric rings (radii 1 and 3, same center),
    spectral clustering with sigma 0.5 recovers the rings
    exactly after the best relabeling: accuracy 1.0. It also
    recovers two plain blobs 10 spreads apart exactly."""
    X, y = make_rings(seed=0)
    labels = SpectralClustering(n_clusters=2).fit_predict(X)
    assert len(np.unique(labels)) == 2
    assert best_accuracy(y, labels, 2) == 1.0

    X, y = make_blobs(seed=0)
    labels = SpectralClustering(n_clusters=2).fit_predict(X)
    assert best_accuracy(y, labels, 2) == 1.0


def test_breaks():
    """Two blobs 10 spreads apart come back exactly at sigma
    0.5. Shrink sigma to 0.01 and the same code on the same
    data falls to near chance: a pair of points 0.1 apart
    already has kernel weight near zero, so almost no edge
    survives, the graph is a pile of isolated points, and the
    smallest eigenvectors carry no blob structure. Accuracy
    drops under 0.75 while the data never changed."""
    X, y = make_blobs(seed=0)
    good = SpectralClustering(n_clusters=2, sigma=0.5).fit(X)
    assert best_accuracy(y, good.labels_, 2) == 1.0

    bad = SpectralClustering(n_clusters=2, sigma=0.01).fit(X)
    acc = best_accuracy(y, bad.labels_, 2)
    assert acc < 0.75


def test_rejects_bad_input():
    """A non-positive sigma, fewer than one cluster, and asking
    for more clusters than points are user errors, not data,
    so they must raise."""
    X, y = make_blobs(seed=0)
    try:
        SpectralClustering(sigma=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for sigma 0")
    try:
        SpectralClustering(n_clusters=0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError for n_clusters 0")
    try:
        SpectralClustering(n_clusters=1000).fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError for too many clusters")
