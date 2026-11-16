"""Tests for Elastic Net: the mixed penalty recovers a sparse
truth and shares a group effect across near-copy features,
where the lasso alone splits it unevenly, but it breaks when a
feature is recorded in much smaller units than the rest,
because the penalty prices the coefficient and not the effect,
so a real signal that would need a large coefficient is set
to exactly zero."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_elastic_net",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
ElasticNet = _impl.ElasticNet
make_grouped = _impl.make_grouped
make_tiny_units = _impl.make_tiny_units


def test_works():
    """On grouped data, elastic net at alpha 0.1 and l1_ratio
    0.5 sets every noise feature to exactly zero, recovers the
    solo weight closely, splits the group effect almost evenly
    (spread under 0.1 across the three near copies), and scores
    above 0.97 R2 on held-out data. The lasso on the same data
    (l1_ratio 1.0) splits the same group with a spread above
    0.3."""
    X, y, w_true = make_grouped(seed=0)
    X_test, y_test, _ = make_grouped(seed=99)
    model = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
    assert np.all(model.coef_[w_true == 0.0] == 0.0)
    assert abs(model.coef_[3] - 2.0) < 0.25
    group = model.coef_[:3]
    assert np.allclose(group, 1.5, atol=0.25)
    assert np.max(group) - np.min(group) < 0.1
    assert abs(model.intercept_ - 1.0) < 0.2
    assert model.score(X_test, y_test) > 0.97

    lasso = ElasticNet(alpha=0.1, l1_ratio=1.0).fit(X, y)
    lasso_group = lasso.coef_[:3]
    assert np.max(lasso_group) - np.min(lasso_group) > 0.3


def test_breaks():
    """The only real feature is the signal divided by 100, so
    its true coefficient is 300. The penalty shrinks on
    coefficient size, so the fit sets that coefficient to
    exactly zero and held-out R2 stays under 0.1: the model
    finds nothing at all. Multiply the same column back up by
    100, changing nothing about the information it carries,
    and the same settings recover a coefficient near 3.0 with
    held-out R2 above 0.95. Nothing about the data changed,
    only the units it was written down in."""
    X, y, w_true = make_tiny_units(seed=0)
    X_test, y_test, _ = make_tiny_units(seed=99)
    model = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
    assert model.coef_[0] == 0.0
    assert model.score(X_test, y_test) < 0.1

    X_big = X.copy()
    X_big[:, 0] *= 100.0
    X_test_big = X_test.copy()
    X_test_big[:, 0] *= 100.0
    rescaled = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_big, y)
    assert abs(rescaled.coef_[0] - 3.0) < 0.3
    assert rescaled.score(X_test_big, y_test) > 0.95


def test_rejects_bad_input():
    """An l1_ratio outside 0 to 1 and a length mismatch between
    X and y are user errors, not data, so they must raise."""
    X, y, _ = make_grouped(seed=0)
    try:
        ElasticNet(l1_ratio=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for l1_ratio above 1")
    try:
        ElasticNet().fit(X, y[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched lengths")
