"""Quadratic Discriminant Analysis: fit one Gaussian per class,
each with its own covariance, and classify by the log posterior.

LDA pools every class into one shared covariance, so its boundary
between two classes is always a straight line. QDA keeps a separate
mean and covariance for each class. The log posterior for class k is
-0.5 * log|Sigma_k| - 0.5 * (x - mu_k)^T Sigma_k^-1 (x - mu_k)
+ log pi_k, and the quadratic terms no longer cancel between
classes, so the boundary can curve. That curve is exactly what lets
QDA separate classes that share a mean and differ only in spread,
the case LDA cannot see. The cost is parameters: each class needs
its own full covariance, so with few points per class the estimates
turn to noise and the curved boundary fits the training set while
failing on new data (numpy only).
Run: python algorithms/qda/impl.py
"""

import numpy as np


def make_same_mean(seed=0, n_per_class=120, d=2):
    """Two classes with the exact same mean and different spreads.

    Class 0 is a tight ball near the origin. Class 1 is a wider
    ball centered at the same point. The means are identical, so
    no straight boundary helps, but the spreads differ, so a
    per-class covariance can tell them apart by distance from the
    center. Returns (X, y).
    """
    rng = np.random.default_rng(seed)
    inner = rng.normal(0.0, 0.5, (n_per_class, d))
    outer = rng.normal(0.0, 2.5, (n_per_class, d))
    X = np.vstack([inner, outer])
    y = np.repeat(np.arange(2), n_per_class)
    return X, y


def make_tiny_sample(seed=0, n_train=10, n_test=400, d=10):
    """Two classes in 10 dimensions with very few training points.

    The classes truly share a spherical covariance and differ only
    in the first coordinate of the mean, so the honest boundary is
    simple. Training gets only n_train points per class, far too
    few to estimate a full 10 by 10 covariance per class, while the
    test set is large. Returns (X_train, y_train, X_test, y_test).
    """
    rng = np.random.default_rng(seed)
    mean0 = np.zeros(d)
    mean1 = np.zeros(d)
    mean1[0] = 2.0
    X_train = np.vstack(
        [
            rng.normal(mean0, 1.0, (n_train, d)),
            rng.normal(mean1, 1.0, (n_train, d)),
        ]
    )
    y_train = np.repeat(np.arange(2), n_train)
    X_test = np.vstack(
        [
            rng.normal(mean0, 1.0, (n_test, d)),
            rng.normal(mean1, 1.0, (n_test, d)),
        ]
    )
    y_test = np.repeat(np.arange(2), n_test)
    return X_train, y_train, X_test, y_test


class QDA:
    """Quadratic Discriminant Analysis with per-class covariances.

    fit stores one mean, one covariance, and one prior per class.
    predict returns the class with the highest Gaussian log
    posterior, including the log determinant term that rewards a
    class whose spread matches the point. A small `reg` ridge on
    each covariance keeps the inverse stable on collinear data.
    """

    def __init__(self, reg=1e-6):
        self.reg = reg

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        self.classes_ = classes
        d = X.shape[1]
        means = []
        covs = []
        priors = []
        logdets = []
        for c in classes:
            Xc = X[y == c]
            mean_c = Xc.mean(axis=0)
            centered = Xc - mean_c
            denom = max(Xc.shape[0] - 1, 1)
            cov_c = centered.T @ centered / denom + self.reg * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov_c)
            if sign <= 0:
                cov_c = cov_c + 1e-4 * np.eye(d)
                sign, logdet = np.linalg.slogdet(cov_c)
            means.append(mean_c)
            covs.append(cov_c)
            priors.append(Xc.shape[0] / X.shape[0])
            logdets.append(logdet)
        self.means_ = np.array(means)
        self.covs_ = np.array(covs)
        self.priors_ = np.array(priors)
        self.logdets_ = np.array(logdets)
        return self

    def log_posterior(self, X):
        """Log posterior score of each row of X under every class."""
        X = np.asarray(X, dtype=float)
        scores = []
        for k in range(len(self.classes_)):
            inv = np.linalg.pinv(self.covs_[k])
            diff = X - self.means_[k]
            maha = np.sum((diff @ inv) * diff, axis=1)
            score = -0.5 * self.logdets_[k] - 0.5 * maha + np.log(self.priors_[k])
            scores.append(score)
        return np.array(scores).T

    def predict(self, X):
        """Assign each row of X to its highest scoring class."""
        return self.classes_[np.argmax(self.log_posterior(X), axis=1)]

    def score(self, X, y):
        """Plain accuracy of predict on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


if __name__ == "__main__":
    print("QDA on two classes with the same mean (seed 0)")
    X, y = make_same_mean(seed=0)
    model = QDA().fit(X, y)
    print(f"class spreads (trace of covariance): "
          f"{np.round([np.trace(c) for c in model.covs_], 2)}")
    print(f"training accuracy on same-mean data: {model.score(X, y):.3f}")
    print("LDA stays near chance here because the means coincide,")
    print("but the per-class covariances differ, so QDA can curve")
    print("its boundary around the tight class")

    print("\nQDA on 10 points per class in 10 dimensions (seed 0)")
    Xtr, ytr, Xte, yte = make_tiny_sample(seed=0)
    small = QDA().fit(Xtr, ytr)
    print(f"training accuracy: {small.score(Xtr, ytr):.3f}")
    print(f"test accuracy: {small.score(Xte, yte):.3f}")
    print("each class covariance has 55 free numbers estimated from")
    print("10 points, so the curved boundary memorizes the training")
    print("set and lands near chance on new data")
