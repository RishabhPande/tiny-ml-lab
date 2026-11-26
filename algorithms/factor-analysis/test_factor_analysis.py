"""Tests for factor analysis: with the right factor count it
rebuilds six variables from two hidden factors and puts the
per-variable noise variance near its true value, and with one
factor on two-factor data it leaves about half of the shared
signal unexplained.
"""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_factor_analysis", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
FactorAnalysis = _impl.FactorAnalysis
make_factor_data = _impl.make_factor_data
reconstruction_r2 = _impl.reconstruction_r2


def test_works():
    """On data built from two factors plus noise with variance
    0.09 per variable, a two-factor fit rebuilds the data with
    an R2 above 0.85, and every estimated unique variance lands
    within 0.06 of the true 0.09, so the private noise is
    measured, not absorbed into the factors."""
    X, Z, W_true = make_factor_data(seed=0)
    model = FactorAnalysis(n_factors=2)
    model.fit(X)

    assert reconstruction_r2(X, model.reconstruct(X)) > 0.85
    assert np.all(np.abs(model.noise_variance_ - 0.09) < 0.06)
    assert model.transform(X).shape == Z.shape


def test_breaks():
    """With one factor on the same two-factor data, the
    reconstruction R2 stays below 0.65 where the two-factor fit
    scores above 0.85. The two true factors are independent and
    equally strong, and a single factor can follow only one
    direction at a time, so the other factor's share of the
    signal has nowhere to go. Adding data does not help; only
    asking for the second factor does."""
    X, Z, W_true = make_factor_data(seed=0)

    one = FactorAnalysis(n_factors=1)
    one.fit(X)
    assert reconstruction_r2(X, one.reconstruct(X)) < 0.65

    two = FactorAnalysis(n_factors=2)
    two.fit(X)
    assert reconstruction_r2(X, two.reconstruct(X)) > 0.85
