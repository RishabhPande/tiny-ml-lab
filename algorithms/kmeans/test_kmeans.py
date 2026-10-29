"""Tests for k-means: it finds blobs, and it fails on rings."""

import os
import sys

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_kmeans", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
KMeans = _impl.KMeans


def _best_accuracy(true, pred):
    """Best accuracy over label permutations (works for 2 clusters)."""
    acc = np.mean(true == pred)
    return max(acc, 1.0 - acc)


def test_works():
    """On round blobs, every true center has a found center nearby."""
    rng = np.random.default_rng(0)
    true_centers = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 4.0]])
    X = np.vstack([c + rng.normal(0, 0.7, (60, 2)) for c in true_centers])
    km = KMeans(n_clusters=3, seed=0).fit(X)
    for c in true_centers:
        dist = np.min(np.linalg.norm(km.centers_ - c, axis=1))
        assert dist < 0.75


def test_breaks():
    """On concentric rings, k-means splits by angle, not by ring."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 200)
    inner = np.column_stack([np.cos(theta[:100]), np.sin(theta[:100])])
    inner += rng.normal(0, 0.1, inner.shape)
    outer = 4.0 * np.column_stack([np.cos(theta[100:]), np.sin(theta[100:])])
    outer += rng.normal(0, 0.1, outer.shape)
    X = np.vstack([inner, outer])
    true = np.array([0] * 100 + [1] * 100)
    km = KMeans(n_clusters=2, seed=0).fit(X)
    assert _best_accuracy(true, km.labels_) < 0.75
