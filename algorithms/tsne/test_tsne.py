"""Tests for t-SNE: the map keeps local neighborhoods, so a
nearest neighbor vote in the embedding recovers the blobs, but
it breaks as a picture of global geometry, because the spacing
between clusters on the map collapses to near equal no matter
how uneven the true spacing is.
"""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_tsne",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
TSNE = _impl.TSNE
make_blobs = _impl.make_blobs
make_uneven_blobs = _impl.make_uneven_blobs
knn_accuracy = _impl._knn_accuracy
centroid_distance_ratio = _impl._centroid_distance_ratio


def test_works():
    """Three blobs many spreads apart in 8 dimensions keep
    their neighborhoods on the map: leave one out 5 nearest
    neighbor accuracy in the 2D embedding is at least 0.95,
    matching the raw data, and the embedding has the right
    shape, one 2D point per input point."""
    X, y = make_blobs(seed=0)
    assert knn_accuracy(X, y) == 1.0
    Y = TSNE(seed=0).fit_transform(X)
    assert Y.shape == (X.shape[0], 2)
    assert knn_accuracy(Y, y) >= 0.95


def test_breaks():
    """Blobs A and B sit 3 units apart and C sits 30 units
    from A, so the raw centroid distance ratio is above 5.
    On the t-SNE map the same ratio falls under 2: the method
    only ever promised to keep neighbors local, so it spreads
    the clusters out near evenly and the true 10 to 1 spacing
    is gone. Anyone reading cluster distances off the plot is
    reading a number the map invented."""
    X, y = make_uneven_blobs(seed=0)
    raw = centroid_distance_ratio(X, y)
    assert raw > 5.0
    Y = TSNE(seed=0).fit_transform(X)
    mapped = centroid_distance_ratio(Y, y)
    assert mapped < 2.0
    assert mapped < raw / 3.0


def test_rejects_bad_input():
    """A perplexity below 1, a perplexity at or above the
    number of points, and fewer than 3 points are user errors,
    not data, so they must raise."""
    X, y = make_blobs(seed=0)
    try:
        TSNE(perplexity=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for perplexity 0.5")
    try:
        TSNE(perplexity=200.0).fit_transform(X)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "expected ValueError for perplexity above n points")
    try:
        TSNE().fit_transform(X[:2])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 2 points")
