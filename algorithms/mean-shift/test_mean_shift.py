"""Tests for mean shift: with a bandwidth matched to the blob
size it finds both blobs and labels them well, and with a
bandwidth wider than the gap between the blobs every walk
climbs to one shared peak, so two clusters come back as one.
"""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_mean_shift", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
MeanShift = _impl.MeanShift
make_blobs = _impl.make_blobs
cluster_accuracy = _impl.cluster_accuracy


def test_works():
    """On two blobs 6 units apart with spread 0.6, a bandwidth
    of 2.0 finds exactly 2 clusters and labels at least 95
    percent of the points correctly. The peaks it finds sit
    close to the true blob centres."""
    X, y, centres = make_blobs(seed=0)
    model = MeanShift(bandwidth=2.0, seed=0)
    labels = model.fit_predict(X)

    assert model.n_clusters_ == 2
    assert cluster_accuracy(y, labels) > 0.95
    found = model.cluster_centers_[np.argsort(model.cluster_centers_[:, 0])]
    assert np.allclose(found, centres, atol=0.5)


def test_breaks():
    """With bandwidth 8.0, wider than the 6 unit gap between
    the blobs, the method finds a single cluster on the same
    data where bandwidth 2.0 finds two. The window covers both
    blobs at once, so the weighted centre always sits between
    them and every walk ends at the same peak. No step count
    fixes this; only a smaller bandwidth does."""
    X, y, centres = make_blobs(seed=0)

    wide = MeanShift(bandwidth=8.0, seed=0)
    wide.fit(X)
    assert wide.n_clusters_ == 1

    matched = MeanShift(bandwidth=2.0, seed=0)
    matched.fit(X)
    assert matched.n_clusters_ == 2
