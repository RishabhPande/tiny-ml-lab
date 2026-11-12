"""Tests for the perceptron: it converges to zero mistakes on
separable data, but on XOR it never converges, because no line
separates opposite corners from the other two."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_perceptron", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
Perceptron = _impl.Perceptron
make_blobs = _impl.make_blobs
make_xor = _impl.make_xor
add_product_feature = _impl.add_product_feature


def test_works():
    """On separable blobs the perceptron reaches an epoch with
    zero mistakes, holds 100 percent train accuracy, and stops
    early instead of burning all 100 epochs."""
    X, y = make_blobs(seed=0)
    model = Perceptron(seed=0)
    history = model.fit(X, y)
    assert model.converged
    assert history[-1] == 0
    assert len(history) < 100
    assert model.score(X, y) == 1.0
    # Held-out points from the same blobs are classified just as well.
    X_new, y_new = make_blobs(seed=1)
    assert model.score(X_new, y_new) > 0.95


def test_breaks():
    """On XOR the perceptron runs all 50 epochs without a single
    clean sweep and stays well under perfect accuracy, because no
    line separates the classes. Adding the x1 * x2 product feature
    makes the same data separable and the same rule converges, so
    the failure is the representation, not the training."""
    X, y = make_xor(seed=0)
    model = Perceptron(seed=0, max_epochs=50)
    history = model.fit(X, y)
    assert not model.converged
    assert len(history) == 50
    assert min(history) > 0
    assert model.score(X, y) < 0.90

    X2 = add_product_feature(X)
    fixed = Perceptron(seed=0)
    fixed.fit(X2, y)
    assert fixed.converged
    assert fixed.score(X2, y) == 1.0
