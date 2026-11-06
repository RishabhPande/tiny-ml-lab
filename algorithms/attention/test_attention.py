"""Tests for attention: weights sum to 1 and a query finds its twin,
but identical keys collapse attention to a uniform mean."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_attention", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
SingleHeadAttention = _impl.SingleHeadAttention
_demo_tokens = _impl._demo_tokens


def _identity_attention(d_model=16):
    """Attention with identity projections, so Q = K = V = X."""
    attn = SingleHeadAttention(d_model=d_model, seed=0)
    attn.W_q = np.eye(d_model)
    attn.W_k = np.eye(d_model)
    attn.W_v = np.eye(d_model)
    attn.W_o = np.eye(d_model)
    return attn


def test_works():
    """Attention weights are proper probabilities (rows sum to 1),
    and a query retrieves the one token that matches it."""
    rng = np.random.default_rng(0)
    attn = SingleHeadAttention(d_model=16, seed=0)
    X = rng.normal(0.0, 1.0, (6, 16))
    out, weights = attn.forward(X)
    assert out.shape == X.shape
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(weights >= 0.0)

    attn_id = _identity_attention(d_model=16)
    query, X = _demo_tokens(rng, n_tokens=6, d_model=16)
    out, weights = attn_id.attend_to(query, X)
    assert out.shape == (16,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-9)
    assert int(np.argmax(weights)) == 4
    assert weights[4] > 0.5
    # The output for the query is mostly its match's value.
    assert np.linalg.norm(out - X[4]) < np.linalg.norm(out - X[1])


def test_breaks():
    """When every token is identical, every key matches every query
    equally: the weights go uniform (1/n each) and every token's
    output is the same mean value, so attention says nothing at all
    about which token is which."""
    rng = np.random.default_rng(0)
    attn = _identity_attention(d_model=16)
    _, X = _demo_tokens(rng, n_tokens=6, d_model=16)
    X_flat = np.tile(X[0], (6, 1))
    out, weights = attn.forward(X_flat)
    assert np.allclose(weights, 1.0 / 6, atol=1e-9)
    assert np.allclose(out, np.tile(X[0], (6, 1)), atol=1e-9)
    # Same collapse with learned (random) weights, not just identity.
    attn_rand = SingleHeadAttention(d_model=16, seed=0)
    _, weights_rand = attn_rand.forward(X_flat)
    assert np.allclose(weights_rand, 1.0 / 6, atol=1e-9)
