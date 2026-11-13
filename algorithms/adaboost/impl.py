"""AdaBoost: boosting decision stumps by reweighting mistakes

AdaBoost (Freund and Schapire, 1997) builds a strong classifier
out of weak ones, here single-split decision stumps. Each round
fits a stump to the data weighted by how badly the ensemble is
doing, gives that stump a vote proportional to its accuracy, and
then raises the weights of the points it still gets wrong. On
clean, separable data the votes add up to a near perfect rule.
On data with many flipped labels the same reweighting turns
against it: the noise points are the hardest, so they collect
the most weight, and later stumps bend toward the noise instead
of the signal (numpy only).
Run: python algorithms/adaboost/impl.py
"""

import numpy as np


def make_blobs(seed=0, n_per_class=100, sep=1.5):
    """Two Gaussian blobs a good boundary can separate.

    Centers sit at (-sep, -sep) and (+sep, +sep) with unit spread,
    labels in {-1, +1}.
    """
    rng = np.random.default_rng(seed)
    X_neg = rng.normal(-sep, 1.0, (n_per_class, 2))
    X_pos = rng.normal(sep, 1.0, (n_per_class, 2))
    X = np.vstack([X_neg, X_pos])
    y = np.concatenate([-np.ones(n_per_class), np.ones(n_per_class)])
    return X, y.astype(int)


def flip_labels(y, frac=0.3, seed=3):
    """Return a copy of y with a fixed-seed fraction flipped."""
    rng = np.random.default_rng(seed)
    y = np.array(y, copy=True)
    n_flip = int(frac * len(y))
    idx = rng.choice(len(y), size=n_flip, replace=False)
    y[idx] = -y[idx]
    return y


def _fit_stump(X, y, w):
    """Best decision stump under sample weights w.

    Tries every feature, up to 20 thresholds per feature, and both
    polarities, and returns (error, feature, threshold, polarity)
    for the stump with the lowest weighted error.
    """
    best = None
    for j in range(X.shape[1]):
        vals = np.sort(np.unique(X[:, j]))
        if len(vals) < 2:
            continue
        thr = (vals[:-1] + vals[1:]) / 2.0
        if len(thr) > 20:
            thr = thr[np.linspace(0, len(thr) - 1, 20).astype(int)]
        for t in thr:
            for pol in (1, -1):
                pred = np.where(X[:, j] >= t, pol, -pol)
                err = float(np.sum(w[pred != y]))
                if best is None or err < best[0]:
                    best = (err, j, float(t), pol)
    return best


def _stump_predict(stump, X):
    _, j, t, pol = stump
    return np.where(X[:, j] >= t, pol, -pol)


class AdaBoost:
    """AdaBoost over decision stumps, labels in {-1, +1}.

    Each round fits the best weighted stump, sets its vote to
    0.5 * log((1 - err) / err) scaled by the learning rate, and
    multiplies the weights of misclassified points by exp(alpha)
    and of correct points by exp(-alpha), then renormalizes.
    """

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps = []
        self.alphas = np.array([])
        self.errors = []

    def fit(self, X, y):
        """Train on (X, y) with labels in {-1, +1}. Returns self."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        w = np.full(n, 1.0 / n)
        self.stumps = []
        self.errors = []
        alphas = []
        for _ in range(self.n_estimators):
            stump = _fit_stump(X, y, w)
            err = stump[0]
            clipped = float(np.clip(err, 1e-10, 1.0 - 1e-10))
            alpha = self.learning_rate * 0.5 * np.log(
                (1.0 - clipped) / clipped
            )
            if err >= 0.5:
                if not self.stumps:
                    self.stumps.append(stump)
                    alphas.append(alpha)
                    self.errors.append(err)
                break
            self.stumps.append(stump)
            alphas.append(alpha)
            self.errors.append(err)
            pred = _stump_predict(stump, X)
            w = w * np.exp(-alpha * y * pred)
            w = w / w.sum()
        self.alphas = np.array(alphas)
        return self

    def decision(self, X):
        """Weighted vote of the stumps for each row of X."""
        X = np.asarray(X, dtype=float)
        scores = np.zeros(X.shape[0])
        for stump, alpha in zip(self.stumps, self.alphas):
            scores = scores + alpha * _stump_predict(stump, X)
        return scores

    def predict(self, X):
        """Predicted labels in {-1, +1}: the sign of the vote."""
        return np.where(self.decision(X) >= 0, 1, -1)

    def score(self, X, y):
        """Accuracy of the ensemble on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


if __name__ == "__main__":
    print("AdaBoost (seed 0)")

    # The works case: clean, separable blobs.
    X, y = make_blobs(seed=0)
    X_test, y_test = make_blobs(seed=99)
    model = AdaBoost(n_estimators=50).fit(X, y)
    print(f"clean blobs: {len(model.stumps)} stumps, "
          f"train accuracy {model.score(X, y):.3f}, "
          f"held-out accuracy {model.score(X_test, y_test):.3f}")

    # A single stump on the same data, for scale.
    stump_only = AdaBoost(n_estimators=1).fit(X, y)
    print(f"one stump only: held-out accuracy "
          f"{stump_only.score(X_test, y_test):.3f}")

    # The break case: 30 percent of the training labels flipped.
    y_noisy = flip_labels(y, frac=0.3, seed=3)
    noisy = AdaBoost(n_estimators=50).fit(X, y_noisy)
    noisy_stump = AdaBoost(n_estimators=1).fit(X, y_noisy)
    print(f"30 percent label noise: boosted held-out accuracy "
          f"{noisy.score(X_test, y_test):.3f}, "
          f"one stump on the same noisy data "
          f"{noisy_stump.score(X_test, y_test):.3f}")
    print("Boosting chases the flipped labels and ends up worse "
          "than the single stump it started from.")
