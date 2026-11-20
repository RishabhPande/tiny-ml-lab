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
    """An extreme bandwidth erases every edge, not just accuracy.

    With no edges the normalized Laplacian is the identity. Every
    eigenvalue is repeated, so its eigenvectors contain no grouping
    information. A particular eigensolver can still happen to return
    labels correlated with the input order; low accuracy is not a
    portable test of this failure mode.
    """
    X, y = make_blobs(seed=0)
    good = SpectralClustering(n_clusters=2, sigma=0.5).fit(X)
    assert best_accuracy(y, good.labels_, 2) == 1.0

    # Check original and shuffled rows: neither graph retains an edge.
    for order in (np.arange(len(X)), np.random.default_rng(7).permutation(len(X))):
        W = _impl._affinity(X[order], sigma=0.0001)
        assert np.count_nonzero(W) == 0
        degree = W.sum(axis=1)
        inv_sqrt = np.where(degree > 0.0,
                            1.0 / np.sqrt(np.maximum(degree, 1e-300)), 0.0)
        L = np.eye(len(X)) - inv_sqrt[:, None] * W * inv_sqrt[None, :]
        np.testing.assert_array_equal(np.linalg.eigvalsh(L), np.ones(len(X)))


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
