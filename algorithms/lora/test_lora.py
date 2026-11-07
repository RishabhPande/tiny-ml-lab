"""Tests for LoRA: a rank-4 adapter fits a rank-2 task while the
frozen layer stays frozen, but a rank-1 adapter cannot fix a
full-rank update."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_lora", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
LoRALinear = _impl.LoRALinear
make_task = _impl.make_task


def test_works():
    """A rank-4 adapter fits a rank-2 task almost exactly, the frozen
    weights never move, the adapter is silent at init, and it trains
    far fewer parameters than a full fine-tune."""
    W0, X, Y, dW_true = make_task(seed=0, rank=2)
    model = LoRALinear(W0, r=4, alpha=1.0, seed=0)
    W0_before = model.W0.copy()

    # At init B is zero, so the adapter adds nothing.
    assert np.allclose(model.forward(X), X @ W0.T, atol=1e-12)

    initial = model.loss(X, Y)
    model.train(X, Y, lr=0.1, steps=500)
    final = model.loss(X, Y)
    assert final < 0.01 * initial
    assert final < 0.01

    # The frozen layer really is frozen.
    assert np.array_equal(model.W0, W0_before)

    # The merged weight recovers the true update.
    merged_err = float(np.mean((model.merged_weight() - (W0 + dW_true)) ** 2))
    assert merged_err < 1e-4

    # Parameter efficiency: 4 * (16 + 8) = 96 trained, not 128.
    assert model.trainable_params() == 96
    assert model.trainable_params() < model.full_params()


def test_breaks():
    """When the true update is full rank, a rank-1 adapter stalls:
    most of the frozen loss remains, because a rank-1 product can
    only point along one direction. The same task with rank 8 fits,
    so the failure is the rank bottleneck, not the training."""
    W0, X, Y, _ = make_task(seed=0, full_rank=True)

    low = LoRALinear(W0, r=1, alpha=1.0, seed=0)
    initial = low.loss(X, Y)
    low.train(X, Y, lr=0.1, steps=500)
    low_final = low.loss(X, Y)
    assert low_final > 0.5 * initial
    # The learned update really is rank 1.
    assert np.linalg.matrix_rank(low.merged_weight() - W0) == 1

    wide = LoRALinear(W0, r=8, alpha=1.0, seed=0)
    wide_initial = wide.loss(X, Y)
    wide.train(X, Y, lr=0.1, steps=500)
    wide_final = wide.loss(X, Y)
    assert wide_final < 0.05 * wide_initial
