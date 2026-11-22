"""Tests for FastICA: it recovers non-Gaussian sources from
linear mixtures, and it has nothing to aim at when the sources
are Gaussian, because every rotation of them is equally valid."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_ica", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
FastICA = _impl.FastICA
make_mixed_signals = _impl.make_mixed_signals
make_mixed_gaussians = _impl.make_mixed_gaussians
matched_correlation = _impl.matched_correlation


def test_works():
    """On mixtures of a sine wave, a square wave, and Laplace
    noise, FastICA recovers the true sources almost exactly, once
    each source is matched to its closest component to allow for
    the order, sign, and scale the method cannot fix."""
    X, S, A = make_mixed_signals(seed=0)
    model = FastICA(seed=0)
    est = model.fit_transform(X)

    assert est.shape == S.shape
    assert matched_correlation(S, est) > 0.95

    # the mixtures really are mixtures: no single observed channel
    # is already one of the sources
    assert matched_correlation(S, X) < 0.9

    # whitened output components are near unit variance and the
    # unmixing is stable on a second call with the fitted model
    est_again = model.transform(X)
    assert np.allclose(est, est_again)
    assert np.allclose(est.std(axis=0), 1.0, atol=0.1)


def test_breaks():
    """With Gaussian sources the same pipeline lands far from the
    true sources. A rotation of independent Gaussians is still a
    set of independent Gaussians, so the non-Gaussianity FastICA
    maximizes is absent and the rotation it returns is arbitrary.
    The failure is identifiability, not tuning: the non-Gaussian
    setup above recovers its sources with the same settings."""
    X, S, A = make_mixed_gaussians(seed=0)
    est = FastICA(seed=0).fit_transform(X)

    gauss_corr = matched_correlation(S, est)
    assert gauss_corr < 0.85

    X_ng, S_ng, _ = make_mixed_signals(seed=0)
    est_ng = FastICA(seed=0).fit_transform(X_ng)
    assert matched_correlation(S_ng, est_ng) > gauss_corr + 0.15
