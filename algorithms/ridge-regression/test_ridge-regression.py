"""Tests for Ridge Regression: a moderate penalty recovers the
true weights on clean data and keeps coefficients small and
stable when features are near copies of each other, but a huge
penalty breaks it, because the shrinkage wins over the data and
the coefficients collapse toward zero until nothing is fit."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_ridge_regression",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
Ridge = _impl.Ridge
make_regression = _impl.make_regression
make_collinear = _impl.make_collinear


def test_works():
    """On clean data, ridge with alpha 1 recovers the true
    weights closely and scores above 0.98 R2 on held-out data.
    On collinear data, ridge at alpha 10 keeps a coefficient
    norm under half of least squares while giving up almost
    nothing in held-out R2."""
    X, y, w_true = make_regression(seed=0)
    X_test, y_test, _ = make_regression(seed=99)
    model = Ridge(alpha=1.0).fit(X, y)
    assert np.allclose(model.coef_, w_true, atol=0.2)
    assert abs(model.intercept_ - 2.0) < 0.2
    assert model.score(X_test, y_test) > 0.98

    X, y, _ = make_collinear(seed=0)
    X_test, y_test, _ = make_collinear(seed=99)
    ols = Ridge(alpha=0.0).fit(X, y)
    ridge = Ridge(alpha=10.0).fit(X, y)
    assert np.linalg.norm(ridge.coef_) < 0.5 * np.linalg.norm(ols.coef_)
    assert ridge.score(X_test, y_test) > 0.90
    assert ridge.score(X_test, y_test) > ols.score(X_test, y_test) - 0.02


def test_breaks():
    """With alpha at 1e6 the penalty dominates: every
    coefficient shrinks to near zero, the model predicts close
    to the mean everywhere, and held-out R2 falls to about
    zero, far below the moderate-penalty model on the same
    data. More regularization is not always safer."""
    X, y, _ = make_regression(seed=0)
    X_test, y_test, _ = make_regression(seed=99)

    good = Ridge(alpha=1.0).fit(X, y)
    crushed = Ridge(alpha=1e6).fit(X, y)

    assert np.all(np.abs(crushed.coef_) < 0.01)
    assert crushed.score(X_test, y_test) < 0.05
    assert good.score(X_test, y_test) > crushed.score(X_test, y_test) + 0.90
