"""Non-Negative Matrix Factorization (NMF): split a non-negative
matrix into two non-negative factors whose product rebuilds it.

Many real matrices are counts or amounts, so every entry is zero
or more: word counts in documents, pixel brightness, purchase
amounts. NMF factors such a matrix X into W and H, both kept
non-negative, so X is close to W @ H. Because nothing can be
subtracted, each column of W reads as a part, a topic, a face
piece, a spending pattern, and each row of H says how much of
that part each sample uses. The Lee and Seung multiplicative
updates move W and H downhill on the squared reconstruction
error while never letting an entry go negative: each update
multiplies by a ratio of positive terms, so a factor that starts
non-negative stays non-negative. The catch is rank. A rank r
factorization can only rebuild matrices whose structure fits in
r parts, and data built from more parts than the model is given
keep a reconstruction error no amount of iterating removes
(numpy only).
Run: python algorithms/nmf/impl.py
"""

import numpy as np


def make_low_rank_data(seed=0, n_samples=60, n_features=40, rank=4):
    """A non-negative matrix built from a known number of parts.

    W_true is n_samples by rank and H_true is rank by
    n_features, both drawn from an exponential distribution so
    they are non-negative and uneven. X is their product plus a
    little non-negative noise, clipped at zero. Returns
    (X, W_true, H_true), so the caller knows the true rank the
    data were built with.
    """
    rng = np.random.default_rng(seed)
    W_true = rng.exponential(scale=1.0, size=(n_samples, rank))
    H_true = rng.exponential(scale=1.0, size=(rank, n_features))
    X = W_true @ H_true
    noise = rng.exponential(scale=0.05, size=X.shape)
    X = X + noise
    return X, W_true, H_true


def relative_error(X, X_hat):
    """Frobenius reconstruction error divided by the size of X.

    0.0 means the factorization rebuilds X exactly. The scale of
    X is divided out, so errors can be compared across datasets.
    """
    X = np.asarray(X, dtype=float)
    X_hat = np.asarray(X_hat, dtype=float)
    return float(np.linalg.norm(X - X_hat) / np.linalg.norm(X))


class NMF:
    """NMF by Lee and Seung multiplicative updates.

    fit learns self.W_ (n_samples by n_components) and self.H_
    (n_components by n_features), both non-negative, with
    W_ @ H_ close to X. The updates alternate: H first, then W,
    each multiplied by the ratio of the data term to the current
    reconstruction term, with a small epsilon in the denominator
    so no division hits zero. Iteration stops early when the
    relative error barely moves.
    """

    def __init__(self, n_components=4, seed=0, max_iter=1000, tol=1e-7):
        self.n_components = n_components
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if np.any(X < 0):
            raise ValueError("NMF needs a non-negative matrix")
        n_samples, n_features = X.shape
        r = self.n_components
        rng = np.random.default_rng(self.seed)
        scale = np.sqrt(X.mean() / r) if X.mean() > 0 else 1.0
        W = rng.random((n_samples, r)) * scale + 0.1 * scale
        H = rng.random((r, n_features)) * scale + 0.1 * scale
        eps = 1e-10
        prev = relative_error(X, W @ H)
        for _ in range(self.max_iter):
            H *= (W.T @ X) / (W.T @ W @ H + eps)
            W *= (X @ H.T) / (W @ H @ H.T + eps)
            err = relative_error(X, W @ H)
            if abs(prev - err) < self.tol:
                prev = err
                break
            prev = err
        self.W_ = W
        self.H_ = H
        self.reconstruction_error_ = prev
        return self

    def transform(self, X):
        """Rebuild X from the fitted factors.

        The factors were fitted to this X, so the reconstruction
        is simply W_ @ H_. X is accepted for API symmetry and
        checked for shape only.
        """
        X = np.asarray(X, dtype=float)
        if X.shape != (self.W_.shape[0], self.H_.shape[1]):
            raise ValueError("X must have the shape used in fit")
        return self.W_ @ self.H_

    def fit_transform(self, X):
        """Fit on X and return (W, H), the two factors."""
        self.fit(X)
        return self.W_, self.H_


if __name__ == "__main__":
    print("NMF with the true rank, 4 parts for rank 4 data (seed 0)")
    X, W_true, H_true = make_low_rank_data(seed=0, rank=4)
    model = NMF(n_components=4, seed=0)
    W, H = model.fit_transform(X)
    print(f"relative reconstruction error: "
          f"{relative_error(X, W @ H):.4f}")
    print(f"factor shapes: W {W.shape}, H {H.shape}, "
          f"both non-negative: {bool(np.all(W >= 0) and np.all(H >= 0))}")
    print("with as many parts as the data were built from, the")
    print("product of the two factors rebuilds the matrix closely")

    print("\nNMF with rank 2 on the same rank 4 data (seed 0)")
    low = NMF(n_components=2, seed=0)
    W2, H2 = low.fit_transform(X)
    print(f"relative reconstruction error: "
          f"{relative_error(X, W2 @ H2):.4f}")
    print("two parts cannot hold four parts worth of structure,")
    print("so a chunk of the matrix is left unrebuilt no matter")
    print("how long the updates run")
