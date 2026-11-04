"""Tests for the Gaussian mixture: one that proves it works, one that
breaks it."""

import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "impl_gmm", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
GaussianMixture = _impl.GaussianMixture


def _two_blobs(seed=42):
    rng = np.random.default_rng(seed)
    left = rng.normal(0, 1.0, (150, 2)) + np.array([-3.0, 0.0])
    right = rng.normal(0, 1.0, (150, 2)) + np.array([3.0, 0.0])
    X = np.vstack([left, right])
    y = np.array([0] * 150 + [1] * 150)
    return X, y


def _overlapping_blobs(seed=7):
    rng = np.random.default_rng(seed)
    first = rng.normal(0, 1.0, (200, 2)) + np.array([0.0, 0.0])
    second = rng.normal(0, 1.0, (200, 2)) + np.array([0.5, 0.0])
    X = np.vstack([first, second])
    y = np.array([0] * 200 + [1] * 200)
    return X, y


def _accuracy_up_to_swap(pred, y):
    straight = np.mean(pred == y)
    swapped = np.mean((1 - pred) == y)
    return max(straight, swapped)


def test_works():
    """On two separated blobs EM must recover centers close to the
    true ones, assign nearly every point to its own blob, and climb
    the log-likelihood without ever going down."""
    X, y = _two_blobs()
    model = GaussianMixture().fit(X)
    assert model.converged_
    assert model.n_iter_ <= 100
    true_centers = np.array([[-3.0, 0.0], [3.0, 0.0]])
    for mean in model.means_:
        assert np.min(np.linalg.norm(true_centers - mean, axis=1)) < 0.5
    assert _accuracy_up_to_swap(model.labels_, y) >= 0.95
    assert np.all(np.abs(model.weights_ - 0.5) < 0.15)
    history = model.log_likelihood_history_
    assert len(history) == model.n_iter_
    assert np.all(np.diff(history) >= -1e-6)
    assert np.allclose(
        model.responsibilities_.sum(axis=1), 1.0, atol=1e-6
    )


def test_breaks():
    """Two blobs 0.5 apart with unit spread overlap almost completely,
    so even the true model cannot tell most points apart. The mixture
    must score well below its separated-blob accuracy here, and its
    average confidence must drop too: soft assignments near 0.5 mean
    the model itself knows it cannot decide."""
    X, y = _overlapping_blobs()
    model = GaussianMixture().fit(X)
    assert _accuracy_up_to_swap(model.labels_, y) < 0.70
    assert np.mean(np.max(model.responsibilities_, axis=1)) < 0.85


def test_rejects_bad_input():
    """Fewer points than components, and non-2D input, are user
    errors, not data, so fit must raise."""
    X, _ = _two_blobs()
    try:
        GaussianMixture(n_components=400).fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for too many components")
    try:
        GaussianMixture().fit(np.zeros(50))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 1D input")
