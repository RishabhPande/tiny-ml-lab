"""Hidden Markov Model (HMM)
Discrete-observation HMM fitted by Baum-Welch (EM), with Viterbi
decoding (numpy only).
Run: python algorithms/hmm/impl.py
"""

import numpy as np


def _normalize_rows(mat):
    """Scale each row of mat to sum to 1. Rows that sum to 0 stay uniform."""
    row_sums = mat.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    out = mat / safe
    empty = (row_sums[:, 0] == 0)
    if np.any(empty):
        out[empty] = 1.0 / mat.shape[1]
    return out


class HiddenMarkovModel:
    """Discrete HMM with n_states hidden states and n_symbols symbols.

    Fitted by Baum-Welch: the E step runs forward-backward to get the
    posterior state occupancies, the M step refits the start, transition,
    and emission probabilities from those occupancies.
    """

    def __init__(self, n_states=2, n_symbols=3, n_iter=100, tol=1e-4,
                 seed=0):
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed

    def _init_params(self, rng):
        self.startprob_ = _normalize_rows(rng.random((1, self.n_states)))[0]
        self.transmat_ = _normalize_rows(rng.random((self.n_states,
                                                     self.n_states)))
        self.emissionprob_ = _normalize_rows(rng.random((self.n_states,
                                                         self.n_symbols)))

    def _forward(self, obs):
        """Scaled forward pass. Returns alpha, scales, log-likelihood."""
        T = len(obs)
        alpha = np.zeros((T, self.n_states))
        alpha[0] = self.startprob_ * self.emissionprob_[:, obs[0]]
        scale = np.zeros(T)
        scale[0] = alpha[0].sum()
        if scale[0] == 0:
            scale[0] = 1.0
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.transmat_) * self.emissionprob_[:, obs[t]]
            scale[t] = alpha[t].sum()
            if scale[t] == 0:
                scale[t] = 1.0
            alpha[t] /= scale[t]
        loglik = float(np.sum(np.log(scale)))
        return alpha, scale, loglik

    def _backward(self, obs, scale):
        """Scaled backward pass, using the forward scales."""
        T = len(obs)
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.transmat_ @ (self.emissionprob_[:, obs[t + 1]] * beta[t + 1]))
            beta[t] /= scale[t + 1]
        return beta

    def score(self, obs):
        """Log-likelihood of an observation sequence under the model."""
        obs = np.asarray(obs, dtype=int)
        _, _, loglik = self._forward(obs)
        return loglik

    def fit(self, obs):
        """Fit by Baum-Welch. Sets startprob_, transmat_, emissionprob_,
        loglik_history_, n_iter_, and converged_."""
        obs = np.asarray(obs, dtype=int)
        rng = np.random.default_rng(self.seed)
        self._init_params(rng)
        T = len(obs)
        self.loglik_history_ = []
        self.converged_ = False
        for it in range(self.n_iter):
            alpha, scale, loglik = self._forward(obs)
            beta = self._backward(obs, scale)
            self.loglik_history_.append(loglik)
            if it > 0 and abs(loglik - self.loglik_history_[-2]) < self.tol:
                self.converged_ = True
                self.n_iter_ = it + 1
                break
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                joint = (alpha[t][:, None] * self.transmat_ *
                         (self.emissionprob_[:, obs[t + 1]] * beta[t + 1])[None, :])
                total = joint.sum()
                xi[t] = joint / total if total > 0 else joint
            self.startprob_ = gamma[0]
            self.transmat_ = _normalize_rows(xi.sum(axis=0))
            emission = np.zeros((self.n_states, self.n_symbols))
            for k in range(self.n_symbols):
                emission[:, k] = gamma[obs == k].sum(axis=0)
            self.emissionprob_ = _normalize_rows(emission)
        else:
            self.n_iter_ = self.n_iter
        return self

    def predict(self, obs):
        """Most likely hidden state sequence (Viterbi decoding)."""
        obs = np.asarray(obs, dtype=int)
        T = len(obs)
        log_start = np.log(np.where(self.startprob_ > 0, self.startprob_, 1e-300))
        log_trans = np.log(np.where(self.transmat_ > 0, self.transmat_, 1e-300))
        log_emit = np.log(np.where(self.emissionprob_ > 0, self.emissionprob_, 1e-300))
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        delta[0] = log_start + log_emit[:, obs[0]]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + log_trans
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(self.n_states)] + log_emit[:, obs[t]]
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def sample(self, n, rng):
        """Generate n observations and their true hidden states."""
        states = np.zeros(n, dtype=int)
        obs = np.zeros(n, dtype=int)
        states[0] = rng.choice(self.n_states, p=self.startprob_)
        obs[0] = rng.choice(self.n_symbols, p=self.emissionprob_[states[0]])
        for t in range(1, n):
            states[t] = rng.choice(self.n_states, p=self.transmat_[states[t - 1]])
            obs[t] = rng.choice(self.n_symbols, p=self.emissionprob_[states[t]])
        return obs, states


def _viterbi_accuracy(true_states, pred_states):
    """Accuracy allowing for label switching (states are unlabeled)."""
    direct = float(np.mean(true_states == pred_states))
    swapped = float(np.mean(true_states == (1 - pred_states)))
    return max(direct, swapped)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true_model = HiddenMarkovModel(n_states=2, n_symbols=3)
    true_model.startprob_ = np.array([0.6, 0.4])
    true_model.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
    true_model.emissionprob_ = np.array([[0.8, 0.15, 0.05],
                                         [0.05, 0.15, 0.8]])
    obs, states = true_model.sample(400, rng)
    model = HiddenMarkovModel(n_states=2, n_symbols=3, n_iter=100, seed=0)
    model.fit(obs)
    print("log-likelihood first / last:",
          round(model.loglik_history_[0], 1), "->",
          round(model.loglik_history_[-1], 1),
          f"({model.n_iter_} iterations)")
    print("Viterbi accuracy:", round(_viterbi_accuracy(states, model.predict(obs)), 3))
    print("learned emission probs:\n", np.round(model.emissionprob_, 2))
    print("learned transition probs:\n", np.round(model.transmat_, 2))
