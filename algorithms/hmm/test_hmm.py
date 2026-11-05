"""Tests for HMM: Baum-Welch recovers hidden states when the emissions
differ, and it cannot when they do not."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_hmm", os.path.join(os.path.dirname(__file__), "impl.py")
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
HiddenMarkovModel = _impl.HiddenMarkovModel
_viterbi_accuracy = _impl._viterbi_accuracy


def _true_model(emission):
    """Two sticky states with the given emission matrix."""
    model = HiddenMarkovModel(n_states=2, n_symbols=3)
    model.startprob_ = np.array([0.6, 0.4])
    model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
    model.emissionprob_ = np.array(emission, dtype=float)
    return model


def test_works():
    """With distinct emissions, Baum-Welch raises the log-likelihood
    and Viterbi recovers the hidden states well."""
    rng = np.random.default_rng(0)
    true_model = _true_model([[0.8, 0.15, 0.05], [0.05, 0.15, 0.8]])
    obs, states = true_model.sample(400, rng)
    model = HiddenMarkovModel(n_states=2, n_symbols=3, n_iter=100, seed=0)
    model.fit(obs)
    assert model.converged_
    assert model.loglik_history_[-1] - model.loglik_history_[0] > 50
    assert _viterbi_accuracy(states, model.predict(obs)) > 0.8
    row_gap = np.max(np.abs(model.emissionprob_[0] - model.emissionprob_[1]))
    assert row_gap > 0.3


def test_breaks():
    """When both states emit the same distribution, the observations
    carry no information about the hidden state: Viterbi decoding
    stays near chance, even though EM still runs and converges."""
    rng = np.random.default_rng(0)
    true_model = _true_model([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
    obs, states = true_model.sample(400, rng)
    model = HiddenMarkovModel(n_states=2, n_symbols=3, n_iter=100, seed=0)
    model.fit(obs)
    assert _viterbi_accuracy(states, model.predict(obs)) < 0.7
