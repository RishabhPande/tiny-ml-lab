"""Tests for AdaBoost: it turns weak stumps into a near perfect
rule on clean data, but label noise breaks it, because the
reweighting keeps pushing later stumps toward the flipped points
until the ensemble is worse than a single stump."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_adaboost", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
AdaBoost = _impl.AdaBoost
make_blobs = _impl.make_blobs
flip_labels = _impl.flip_labels


def test_works():
    """On clean separable blobs, 50 boosted stumps fit the train
    set perfectly and beat a single stump on held-out data."""
    X, y = make_blobs(seed=0)
    X_test, y_test = make_blobs(seed=99)
    model = AdaBoost(n_estimators=50).fit(X, y)
    assert len(model.stumps) == 50
    assert model.score(X, y) == 1.0
    assert model.score(X_test, y_test) > 0.95

    stump_only = AdaBoost(n_estimators=1).fit(X, y)
    assert model.score(X_test, y_test) > stump_only.score(X_test, y_test)


def test_breaks():
    """With 30 percent of the training labels flipped, boosting
    ends up worse on clean held-out data than one stump trained
    on the same noisy data, and far worse than boosting trained
    on clean labels. The flipped points are always the hardest,
    so they collect the weight and bend the later stumps."""
    X, y = make_blobs(seed=0)
    X_test, y_test = make_blobs(seed=99)
    y_noisy = flip_labels(y, frac=0.3, seed=3)
    assert float(np.mean(y_noisy != y)) == 0.3

    clean = AdaBoost(n_estimators=50).fit(X, y)
    noisy = AdaBoost(n_estimators=50).fit(X, y_noisy)
    noisy_stump = AdaBoost(n_estimators=1).fit(X, y_noisy)

    noisy_acc = noisy.score(X_test, y_test)
    assert noisy_acc < 0.90
    assert noisy_acc < noisy_stump.score(X_test, y_test) - 0.05
    assert clean.score(X_test, y_test) > noisy_acc + 0.10
