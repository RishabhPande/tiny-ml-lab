"""Tests for Hierarchical Clustering: agglomerative merging
with single linkage recovers three far apart blobs exactly,
but it breaks when a thin chain of points joins two blobs,
because single linkage only looks at the nearest pair, so a
chain of small steps merges clusters that are far apart.
"""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_hierarchical_clustering",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
AgglomerativeClustering = _impl.AgglomerativeClustering
make_blobs = _impl.make_blobs
make_bridge = _impl.make_bridge


def test_works():
    """On three blobs about 15 spreads apart, single linkage
    cut at three clusters recovers the truth exactly: every
    found cluster holds points from only one true blob, and
    every true blob lands in only one found cluster. Complete
    and average linkage recover the same grouping."""
    X, y = make_blobs(seed=0)
    for linkage in ("single", "complete", "average"):
        labels = AgglomerativeClustering(
            n_clusters=3, linkage=linkage).fit_predict(X)
        assert len(np.unique(labels)) == 3
        for k in range(3):
            assert len(np.unique(y[labels == k])) == 1
            assert len(np.unique(labels[y == k])) == 1


def test_breaks():
    """Two blobs 20 spreads apart, cut by a distance threshold
    of 1.0 under single linkage, give exactly two clusters
    when nothing joins them. Add 13 bridge points spaced
    about 0.4 apart between the blobs and the same settings
    give one cluster: the chain of small steps chains the two
    blobs together, and every blob point shares a label, even
    though the blobs themselves did not move at all."""
    X, y = make_bridge(seed=0)
    blobs_only = X[:40]
    clean = AgglomerativeClustering(
        distance_threshold=1.0, linkage="single").fit(blobs_only)
    assert clean.n_clusters_ == 2
    assert len(np.unique(clean.labels_[:20])) == 1
    assert len(np.unique(clean.labels_[20:])) == 1
    assert clean.labels_[0] != clean.labels_[20]

    bridged = AgglomerativeClustering(
        distance_threshold=1.0, linkage="single").fit(X)
    assert bridged.n_clusters_ == 1
    assert len(np.unique(bridged.labels_)) == 1


def test_rejects_bad_input():
    """An unknown linkage, asking for more clusters than
    points, and giving both stopping rules at once are user
    errors, not data, so they must raise."""
    X, y = make_blobs(seed=0)
    try:
        AgglomerativeClustering(linkage="ward")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad linkage")
    try:
        AgglomerativeClustering(n_clusters=1000).fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError for too many clusters")
    try:
        AgglomerativeClustering(
            n_clusters=3, distance_threshold=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError for both stopping rules")
