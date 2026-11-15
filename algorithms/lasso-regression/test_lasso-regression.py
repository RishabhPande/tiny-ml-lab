"""Tests for Lasso Regression: the L1 penalty recovers a sparse
truth with most coefficients at exactly zero, but it breaks as
a way of reading the data when two features are near copies of
each other, because it hands the whole shared effect to one of
them and zeroes the other, even though the truth splits it."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_lasso_regression",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
Lasso = _impl.Lasso
make_sparse = _impl.make_sparse
make_twins = _impl.make_twins


def test_works():
    """On sparse data, lasso at alpha 0.1 sets every useless
    feature to exactly zero, recovers the three real weights
    closely, and scores above 0.97 R2 on held-out data. Least
    squares on the same data never sets any of them to zero."""
    X, y, w_true = make_sparse(seed=0)
    X_test, y_test, _ = make_sparse(seed=99)
    model = Lasso(alpha=0.1).fit(X, y)
    assert np.all(model.coef_[w_true == 0.0] == 0.0)
    assert np.allclose(model.coef_[w_true != 0.0],
                       w_true[w_true != 0.0], atol=0.25)
    assert abs(model.intercept_ - 2.0) < 0.2
    assert model.score(X_test, y_test) > 0.97

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    ols_coef = np.linalg.solve(Xc.T @ Xc, Xc.T @ yc)
    assert np.all(ols_coef[w_true == 0.0] != 0.0)


def test_breaks():
    """Two features are near copies of one factor and the truth
    splits the effect evenly, (2.0, 2.0). The lasso instead
    loads almost everything onto one twin and sets the other
    to exactly zero. Held-out R2 stays above 0.95, so nothing
    looks wrong until the coefficients are read as an
    explanation: the largest coefficient error is above 1.0
    even though the combined effect is about right."""
    X, y, w_true = make_twins(seed=0)
    X_test, y_test, _ = make_twins(seed=99)
    model = Lasso(alpha=0.1).fit(X, y)
    assert model.score(X_test, y_test) > 0.95
    assert abs(model.coef_.sum() - 4.0) < 0.3
    assert np.min(np.abs(model.coef_)) < 0.05
    assert np.max(np.abs(model.coef_ - w_true)) > 1.0


def test_rejects_bad_input():
    """A negative alpha and a length mismatch between X and y
    are user errors, not data, so they must raise."""
    X, y, _ = make_sparse(seed=0)
    try:
        Lasso(alpha=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative alpha")
    try:
        Lasso().fit(X, y[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched lengths")
