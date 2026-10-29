"""Tests for PCA: it compresses lines, and it cannot see a circle."""

import os
import sys

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_pca", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
PCA = _impl.PCA


def test_works():
    """On elongated data, the first component holds almost all variance,
    and 1-component reconstruction is close."""
    rng = np.random.default_rng(0)
    t = rng.uniform(-5, 5, 200)
    X = np.column_stack([t, 0.2 * t + rng.normal(0, 0.3, 200)])
    pca = PCA(n_components=1).fit(X)
    assert pca.explained_variance_ratio_[0] > 0.9
    recon = pca.inverse_transform(pca.transform(X))
    rel_err = float(np.mean((X - recon) ** 2)) / float(np.mean((X - X.mean(axis=0)) ** 2))
    assert rel_err < 0.2


def test_breaks():
    """On points around a circle, no single line explains the data: the
    first component holds only about half the variance."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 300)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    X += rng.normal(0, 0.05, X.shape)
    pca = PCA(n_components=1).fit(X)
    assert pca.explained_variance_ratio_[0] < 0.7
    assert 1.0 - pca.explained_variance_ratio_[0] > 0.3
