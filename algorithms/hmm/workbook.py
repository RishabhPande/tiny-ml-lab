"""HMM workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def _normalize_rows(mat):
    """TODO: scale each row of mat to sum to 1."""
    raise NotImplementedError("TODO: implement _normalize_rows")


class HiddenMarkovModel:
    """TODO: discrete HMM fitted by Baum-Welch, with Viterbi decoding."""

    def __init__(self, n_states=2, n_symbols=3, n_iter=100, tol=1e-4,
                 seed=0):
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed

    def _init_params(self, rng):
        """TODO: random stochastic start, transition, emission probs."""
        raise NotImplementedError("TODO: implement _init_params")

    def _forward(self, obs):
        """TODO: scaled forward pass. Return alpha, scales, log-likelihood."""
        raise NotImplementedError("TODO: implement _forward")

    def _backward(self, obs, scale):
        """TODO: scaled backward pass, using the forward scales."""
        raise NotImplementedError("TODO: implement _backward")

    def score(self, obs):
        """TODO: log-likelihood of an observation sequence."""
        raise NotImplementedError("TODO: implement score")

    def fit(self, obs):
        """TODO: Baum-Welch. Set startprob_, transmat_, emissionprob_,
        loglik_history_, n_iter_, and converged_."""
        raise NotImplementedError("TODO: implement fit")

    def predict(self, obs):
        """TODO: most likely hidden state sequence (Viterbi decoding)."""
        raise NotImplementedError("TODO: implement predict")

    def sample(self, n, rng):
        """TODO: generate n observations and their true hidden states."""
        raise NotImplementedError("TODO: implement sample")


def _check(name, fn):
    try:
        fn()
    except NotImplementedError:
        print(f"[TODO] {name} is not implemented yet")
        return False
    except Exception as exc:  # noqa: BLE001 - workbook feedback, not a test
        print(f"[ERROR] {name} raised {type(exc).__name__}: {exc}")
        return False
    print(f"[OK] {name} runs")
    return True


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    model = HiddenMarkovModel(n_states=2, n_symbols=3)
    results = [
        _check("_normalize_rows", lambda: _normalize_rows(np.ones((2, 3)))),
        _check("_init_params", lambda: model._init_params(rng)),
        _check("sample", lambda: model.sample(10, rng)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_hmm.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
