"""Tests for QDA: it separates classes that share a mean and
differ in spread, and it falls apart when each class covariance
must be estimated from a handful of points in many dimensions."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_qda", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
QDA = _impl.QDA
make_same_mean = _impl.make_same_mean
make_tiny_sample = _impl.make_tiny_sample


def test_works():
    """On same-mean data where the classes differ only in spread,
    QDA gets high accuracy, because its per-class covariances pick
    up the spread difference that a shared-covariance method and
    any straight boundary miss."""
    X, y = make_same_mean(seed=0)
    model = QDA().fit(X, y)

    acc = model.score(X, y)
    assert acc > 0.85

    traces = [float(np.trace(c)) for c in model.covs_]
    assert traces[1] > 10.0 * traces[0]
    assert model.means_.shape == (2, X.shape[1])
    assert abs(float(np.sum(model.priors_)) - 1.0) < 1e-9


def test_breaks():
    """With 10 training points per class in 10 dimensions, QDA
    fits its training set perfectly and then scores near chance
    on a large test set drawn from the same distribution. The
    failure is estimation, not the data: the same setup trained
    on many more points per class generalizes far better."""
    X_train, y_train, X_test, y_test = make_tiny_sample(seed=0)
    model = QDA().fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    assert train_acc > 0.95
    assert test_acc < 0.65
    assert train_acc > test_acc + 0.30

    X_big, y_big, _, _ = make_tiny_sample(seed=0, n_train=400, n_test=10)
    big = QDA().fit(X_big, y_big)
    big_test_acc = big.score(X_test, y_test)
    assert big_test_acc > test_acc + 0.15
