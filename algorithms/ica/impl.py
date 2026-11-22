"""Independent Component Analysis (FastICA): unmix linearly mixed
signals by looking for the most non-Gaussian directions.

PCA finds directions of large variance and decorrelates the data,
but decorrelated is not the same as independent. ICA goes one step
further: after whitening, it rotates the data until each component
is as non-Gaussian as possible, measured here with a tanh contrast
(the fixed-point FastICA rule of Hyvarinen). The classic picture is
the cocktail party: microphones record mixtures of speakers, and
ICA recovers the speakers, up to order, sign, and scale, which no
method of this family can pin down. The catch is in the assumption:
the rotation is only identifiable because the true sources are
non-Gaussian. Mix Gaussian sources instead and every rotation of
the whitened data looks exactly as Gaussian and exactly as
independent, so there is nothing left to aim at (numpy only).
Run: python algorithms/ica/impl.py
"""

import numpy as np


def make_mixed_signals(seed=0, n=3000):
    """Three non-Gaussian sources, mixed by a fixed matrix.

    The sources are a sine wave, a square wave, and Laplace noise,
    each standardized. They are mixed with a known 3 by 3 matrix,
    so the caller gets the mixtures X, the true sources S, and the
    mixing matrix A, and can score how well ICA unmixes them.
    Returns (X, S, A).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 8.0 * np.pi, n)
    s1 = np.sin(2.0 * t)
    s2 = np.sign(np.sin(3.0 * t + 0.5))
    s3 = rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)
    S = np.column_stack([s1, s2, s3])
    S = (S - S.mean(axis=0)) / S.std(axis=0)
    A = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.5, 2.0, 1.0],
            [1.5, 1.0, 0.5],
        ]
    )
    X = S @ A.T
    return X, S, A


def make_mixed_gaussians(seed=0, n=3000):
    """Three independent Gaussian sources, mixed the same way.

    Same mixing matrix as make_mixed_signals, but every source is
    standard normal. A rotation of independent Gaussians is still a
    set of independent Gaussians, so the mixed data carry no trace
    of the rotation ICA would need to undo. Returns (X, S, A).
    """
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n, 3))
    S = (S - S.mean(axis=0)) / S.std(axis=0)
    A = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.5, 2.0, 1.0],
            [1.5, 1.0, 0.5],
        ]
    )
    X = S @ A.T
    return X, S, A


def matched_correlation(S_true, S_est):
    """Mean best-match absolute correlation between true sources
    and estimated components.

    ICA cannot fix the order, sign, or scale of its components, so
    scoring matches each true source to the still-unmatched
    component with the highest absolute correlation, greedily, and
    averages. 1.0 means the sources were recovered exactly, up to
    those unavoidable ambiguities.
    """
    S_true = np.asarray(S_true, dtype=float)
    S_est = np.asarray(S_est, dtype=float)
    n_true = S_true.shape[1]
    both = np.hstack([S_true, S_est])
    corr = np.abs(np.corrcoef(both, rowvar=False)[:n_true, n_true:])
    used_true = set()
    used_est = set()
    values = []
    flat = sorted(
        (
            (float(corr[i, j]), i, j)
            for i in range(corr.shape[0])
            for j in range(corr.shape[1])
        ),
        reverse=True,
    )
    for value, i, j in flat:
        if i not in used_true and j not in used_est:
            used_true.add(i)
            used_est.add(j)
            values.append(value)
    return float(np.mean(values))


class FastICA:
    """FastICA with whitening, a tanh contrast, and symmetric
    decorrelation of the unmixing rows at every step.

    fit centers and whitens X, then runs the fixed-point iteration
    for all components at once. transform applies the learned
    unmixing to new data. The components come out in no promised
    order, with arbitrary sign and unit variance, which is the
    honest output of the method.
    """

    def __init__(self, seed=0, max_iter=500, tol=1e-6):
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = Xc.T @ Xc / (n - 1)
        vals, vecs = np.linalg.eigh(cov)
        whiten = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        self.whiten_ = whiten
        Z = Xc @ whiten

        rng = np.random.default_rng(self.seed)
        W, _ = np.linalg.qr(rng.normal(size=(d, d)))
        W = W.T
        for _ in range(self.max_iter):
            Y = Z @ W.T
            g = np.tanh(Y)
            gp_mean = np.mean(1.0 - g**2, axis=0)
            W_new = (g.T @ Z) / n - np.diag(gp_mean) @ W
            evals, evecs = np.linalg.eigh(W_new @ W_new.T)
            W_new = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T @ W_new
            delta = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0))
            W = W_new
            if delta < self.tol:
                break
        self.unmixing_ = W
        return self

    def transform(self, X):
        """Apply the learned unmixing to the rows of X."""
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.whiten_ @ self.unmixing_.T

    def fit_transform(self, X):
        """Fit on X and return its estimated source signals."""
        return self.fit(X).transform(X)


if __name__ == "__main__":
    print("FastICA on three mixed non-Gaussian sources (seed 0)")
    X, S, A = make_mixed_signals(seed=0)
    est = FastICA(seed=0).fit_transform(X)
    print(f"matched correlation with the true sources: "
          f"{matched_correlation(S, est):.3f}")
    print("up to order, sign, and scale, the sine, the square wave,")
    print("and the Laplace noise are all back")

    print("\nFastICA on three mixed Gaussian sources (seed 0)")
    Xg, Sg, _ = make_mixed_gaussians(seed=0)
    est_g = FastICA(seed=0).fit_transform(Xg)
    print(f"matched correlation with the true sources: "
          f"{matched_correlation(Sg, est_g):.3f}")
    print("a rotation of independent Gaussians is still independent")
    print("Gaussians, so no rotation stands out and the unmixing")
    print("lands at an arbitrary angle, not at the true sources")
