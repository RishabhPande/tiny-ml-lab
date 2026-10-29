"""Gaussian naive Bayes classifier (numpy only)."""

import numpy as np


class GaussianNB:
    """Gaussian naive Bayes: independent normal features per class."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        """Estimate per-class priors, means, and variances. Returns self."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.priors_ = []
        self.means_ = []
        self.vars_ = []
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_.append(len(Xc) / len(X))
            self.means_.append(Xc.mean(axis=0))
            self.vars_.append(Xc.var(axis=0) + self.var_smoothing)
        self.priors_ = np.array(self.priors_)
        self.means_ = np.array(self.means_)
        self.vars_ = np.array(self.vars_)
        return self

    def _log_posterior(self, X):
        X = np.asarray(X, dtype=float)
        log_post = []
        for i in range(len(self.classes_)):
            log_prior = np.log(self.priors_[i])
            log_lik = -0.5 * np.sum(
                np.log(2 * np.pi * self.vars_[i])
                + (X - self.means_[i]) ** 2 / self.vars_[i],
                axis=1,
            )
            log_post.append(log_prior + log_lik)
        return np.array(log_post).T

    def predict(self, X):
        """Predict one label per row of X."""
        return self.classes_[np.argmax(self._log_posterior(X), axis=1)]

    def predict_proba(self, X):
        """Predicted class probabilities per row of X."""
        log_post = self._log_posterior(X)
        log_post = log_post - log_post.max(axis=1, keepdims=True)
        probs = np.exp(log_post)
        return probs / probs.sum(axis=1, keepdims=True)


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_train = np.vstack(
        [rng.normal(0, 1, (200, 2)), rng.normal(0, 1, (200, 2)) + [3, 3]]
    )
    y_train = np.array([0] * 200 + [1] * 200)
    X_test = np.vstack(
        [rng.normal(0, 1, (100, 2)), rng.normal(0, 1, (100, 2)) + [3, 3]]
    )
    y_test = np.array([0] * 100 + [1] * 100)
    nb = GaussianNB().fit(X_train, y_train)
    print("test accuracy:", round(accuracy(y_test, nb.predict(X_test)), 3))
