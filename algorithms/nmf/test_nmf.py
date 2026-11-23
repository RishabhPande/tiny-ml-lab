"""Tests for NMF: with as many parts as the data were built
from, the two non-negative factors rebuild the matrix closely,
and with too few parts a large chunk of the matrix stays
unrebuilt, because a low rank product cannot hold higher rank
structure."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_nmf", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
NMF = _impl.NMF
make_low_rank_data = _impl.make_low_rank_data
relative_error = _impl.relative_error


def test_works():
    """On data built from 4 parts, a rank 4 NMF rebuilds the
    matrix to under 5 percent relative error, and both factors
    stay non-negative, which is the whole point of the method:
    the parts can only add, never subtract."""
    X, W_true, H_true = make_low_rank_data(seed=0, rank=4)
    model = NMF(n_components=4, seed=0)
    W, H = model.fit_transform(X)

    assert W.shape == (60, 4)
    assert H.shape == (4, 40)
    assert np.all(W >= 0)
    assert np.all(H >= 0)
    assert relative_error(X, W @ H) < 0.05
    assert model.reconstruction_error_ < 0.05

    # the reconstruction helper agrees with the factors, and a
    # plain mean baseline is nowhere near as good
    assert np.allclose(model.transform(X), W @ H)
    baseline = np.full_like(X, X.mean())
    assert relative_error(X, baseline) > 0.5


def test_breaks():
    """With rank 2 on rank 4 data, the relative error stays
    above 15 percent, more than ten times the rank 4 error on
    the same data with the same settings. The missing parts are
    not a tuning problem: a rank 2 product has no room for the
    other two parts, so the error floor is set by the rank the
    caller chose, not by the number of iterations."""
    X, W_true, H_true = make_low_rank_data(seed=0, rank=4)

    low = NMF(n_components=2, seed=0)
    W2, H2 = low.fit_transform(X)
    low_err = relative_error(X, W2 @ H2)
    assert low_err > 0.15

    full = NMF(n_components=4, seed=0)
    W4, H4 = full.fit_transform(X)
    full_err = relative_error(X, W4 @ H4)
    assert low_err > 10 * full_err
