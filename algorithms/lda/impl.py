"""Linear Discriminant Analysis: project onto the directions that
separate class means, relative to the spread inside each class.

LDA builds two scatter matrices. The within-class scatter Sw adds
up how points spread around their own class mean. The between-class
scatter Sb measures how the class means spread around the overall
mean. The best projection directions solve the generalized eigen
problem for inv(Sw) @ Sb: a direction scores high when the class
means land far apart along it and the classes stay tight along it.
Classification then uses the nearest class centroid in the projected
space. Only class means and shared spread enter the math, so two
classes with the same mean are invisible to it (numpy only).
Run: python algorithms/lda/impl.py
"""

import numpy as np


def make_blobs(seed=0, n_per_class=80, d=4):
    """Three well separated Gaussian classes with a fixed seed.

    Returns (X, y) with class means at least 4 units apart and
    unit spread inside each class, so LDA should separate them
    almost perfectly.
    """
    rng = np.random.default_rng(seed)
    means = np.zeros((3, d))
    means[0, 0] = -3.0
    means[1, 0] = 3.0
    means[2, 1] = 3.5
    X = np.vstack(
        [rng.normal(means[k], 0.7, (n_per_class, d)) for k in range(3)]
    )
    y = np.repeat(np.arange(3), n_per_class)
    return X, y


def make_same_mean(seed=0, n_per_class=120, d=2):
    """Two classes with the exact same mean and different spreads.

    Class 0 is a tight ball near the origin. Class 1 is a wider
    ball centered at the same point. A person can separate them
    by distance from the center, but their means are identical,
    so the between-class scatter is near zero and LDA has almost
    nothing to work with. Returns (X, y).
    """
    rng = np.random.default_rng(seed)
    inner = rng.normal(0.0, 0.5, (n_per_class, d))
    outer = rng.normal(0.0, 2.5, (n_per_class, d))
    X = np.vstack([inner, outer])
    y = np.repeat(np.arange(2), n_per_class)
    return X, y


class LDA:
    """Linear Discriminant Analysis with nearest-centroid prediction.

    fit stores the top `n_components` discriminant directions in
    `components_` (shape (n_components, d)) and the class centroids
    in the projected space. predict assigns each point to the
    nearest projected centroid.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        self.classes_ = classes
        d = X.shape[1]
        overall = X.mean(axis=0)
        Sw = np.zeros((d, d))
        Sb = np.zeros((d, d))
        self.means_ = []
        for c in classes:
            Xc = X[y == c]
            mean_c = Xc.mean(axis=0)
            self.means_.append(mean_c)
            centered = Xc - mean_c
            Sw = Sw + centered.T @ centered
            diff = (mean_c - overall).reshape(d, 1)
            Sb = Sb + Xc.shape[0] * (diff @ diff.T)
        self.means_ = np.array(self.means_)
        # Regularize a touch so the inverse is stable on collinear data.
        Sw_reg = Sw + 1e-6 * np.trace(Sw) / d * np.eye(d)
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw_reg) @ Sb)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        order = np.argsort(eigvals)[::-1]
        self.eigenvalues_ = eigvals[order]
        k = min(self.n_components, d, len(classes) - 1 if len(classes) > 1 else 1)
        self.n_components_ = max(k, 1)
        chosen = eigvecs[:, order[: self.n_components_]].T
        norms = np.linalg.norm(chosen, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.components_ = chosen / norms
        self.centroids_ = self.transform(self.means_)
        return self

    def transform(self, X):
        """Project X onto the discriminant directions."""
        X = np.asarray(X, dtype=float)
        return X @ self.components_.T

    def predict(self, X):
        """Assign each row of X to its nearest projected centroid."""
        Z = self.transform(X)
        dists = np.sum((Z[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)
        return self.classes_[np.argmin(dists, axis=1)]

    def score(self, X, y):
        """Plain accuracy of predict on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


if __name__ == "__main__":
    print("LDA on three separated blobs (seed 0)")
    X, y = make_blobs(seed=0)
    model = LDA(n_components=2).fit(X, y)
    print(f"top eigenvalues: {np.round(model.eigenvalues_[:2], 2)}")
    print(f"training accuracy on blobs: {model.score(X, y):.3f}")

    print("\nLDA on two classes with the same mean (seed 0)")
    X2, y2 = make_same_mean(seed=0)
    hard = LDA(n_components=1).fit(X2, y2)
    print(f"top eigenvalue: {hard.eigenvalues_[0]:.4f}")
    print(f"training accuracy on same-mean data: {hard.score(X2, y2):.3f}")
    print("the classes differ only in spread, and LDA only sees means,")
    print("so it stays close to chance even though radius separates them")
    radius = np.linalg.norm(X2, axis=1)
    by_radius = (radius > 1.2).astype(int)
    print(f"accuracy of a simple radius rule: {np.mean(by_radius == y2):.3f}")
