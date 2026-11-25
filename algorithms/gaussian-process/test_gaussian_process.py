"""Tests for Gaussian process regression: with a length scale
matched to the data it tracks a sine wave closely and its
error bars grow away from the data, and with a length scale far
wider than the data range the kernel goes nearly constant and
the prediction flattens toward the training mean.
"""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_gaussian_process", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
GaussianProcessRegressor = _impl.GaussianProcessRegressor
make_wave = _impl.make_wave
r2_score = _impl.r2_score


def test_works():
    """On a sine wave sampled at 24 points with noise 0.1, a
    length scale of 1.0 scores an R2 above 0.95 on a clean test
    grid. The predictive standard deviation at x = 12, far
    outside the training range [0, 6], is larger than the one
    in the middle of the data, which is the uncertainty doing
    its job."""
    X_train, y_train, X_test, y_test = make_wave(seed=0)
    model = GaussianProcessRegressor(length_scale=1.0, noise_std=0.1)
    model.fit(X_train, y_train)
    mean, std = model.predict(X_test)

    assert r2_score(y_test, mean) > 0.95
    _, far_std = model.predict(np.array([[12.0]]))
    assert far_std[0] > std[30]


def test_breaks():
    """With length scale 10.0 on data spanning only [0, 6], the
    method scores an R2 below 0.3 on the same wave where length
    scale 1.0 scores above 0.95. Every distance is tiny next to
    the length scale, so the kernel matrix is nearly constant,
    the weights stop depending on position, and the prediction
    flattens toward the training mean. No extra data fixes a
    kernel that cannot tell near from far."""
    X_train, y_train, X_test, y_test = make_wave(seed=0)

    wide = GaussianProcessRegressor(length_scale=10.0, noise_std=0.1)
    wide.fit(X_train, y_train)
    wide_mean, _ = wide.predict(X_test)
    assert r2_score(y_test, wide_mean) < 0.3

    matched = GaussianProcessRegressor(length_scale=1.0, noise_std=0.1)
    matched.fit(X_train, y_train)
    matched_mean, _ = matched.predict(X_test)
    assert r2_score(y_test, matched_mean) > 0.95
