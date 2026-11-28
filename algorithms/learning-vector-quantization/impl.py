"""Learning Vector Quantization: a small set of labelled
prototypes that learn to stand guard over their class.

Each class gets a few prototype vectors. Training repeats one
simple step: draw a data point, find its closest prototype,
and move that prototype toward the point when the prototype
carries the right label, or push it away when it carries the
wrong one. The step size decays over time, so the prototypes
first move a lot and then settle. Prediction is plain nearest
prototype: a new point takes the label of the closest
prototype, so the prototypes carve the space into labelled
regions. The catch is coverage: a class that lives in two far
apart clusters needs at least one prototype per cluster. With
only one prototype per class, the stranded cluster's points
sit closer to the other class's prototype and get its label,
and no amount of training fixes a prototype that is not there
(numpy only).
Run: python algorithms/learning-vector-quantization/impl.py
"""

import numpy as np


def make_blob_data(seed=0, n_per_blob=120, blob_std=0.4):
    """Three well separated 2D blobs, one per class.

    Centres sit at (0, 0), (4, 0) and (2, 3.5), far apart
    compared with the per-blob spread. Returns (X, labels)
    with points grouped by blob in order.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5]])
    blocks = [
        centre + rng.normal(scale=blob_std, size=(n_per_blob, 2))
        for centre in centres
    ]
    X = np.vstack(blocks)
    labels = np.repeat(np.arange(3), n_per_blob)
    return X, labels


def make_split_class_data(seed=0, n_per_blob=120, blob_std=0.4):
    """Two classes where class 0 lives in two far apart blobs.

    Class 0 sits at (-4, 0) and (4, 0), and class 1 sits in
    between at (0, 0). Returns (X, labels) with class 0 points
    first, then class 1 points.
    """
    rng = np.random.default_rng(seed)
    left = np.array([-4.0, 0.0]) + rng.normal(
        scale=blob_std, size=(n_per_blob, 2)
    )
    right = np.array([4.0, 0.0]) + rng.normal(
        scale=blob_std, size=(n_per_blob, 2)
    )
    middle = np.array([0.0, 0.0]) + rng.normal(
        scale=blob_std, size=(n_per_blob, 2)
    )
    X = np.vstack([left, right, middle])
    labels = np.array([0] * (2 * n_per_blob) + [1] * n_per_blob)
    return X, labels


class LearningVectorQuantization:
    """LVQ1: labelled prototypes pulled toward correct points
    and pushed away from wrong ones.

    Prototypes start as random training points, the right
    number taken from each class. Each step draws one point,
    finds its closest prototype, and moves that prototype by
    the current learning rate, toward the point on a label
    match and away from it on a mismatch. The learning rate
    decays exponentially to 5 percent of its start across the
    run, so late steps only polish.
    """

    def __init__(self, prototypes_per_class=2, learning_rate=0.3,
                 n_iter=3000):
        if prototypes_per_class < 1:
            raise ValueError("each class needs at least one prototype")
        self.prototypes_per_class = prototypes_per_class
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y, seed=0):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        rng = np.random.default_rng(seed)
        classes = np.unique(y)
        starts = []
        proto_labels = []
        for cls in classes:
            pool = X[y == cls]
            pick = rng.choice(
                len(pool),
                size=self.prototypes_per_class,
                replace=self.prototypes_per_class > len(pool),
            )
            starts.append(pool[pick])
            proto_labels.extend([cls] * self.prototypes_per_class)
        prototypes = np.vstack(starts) + rng.normal(
            scale=0.01, size=(len(proto_labels), X.shape[1])
        )
        self.prototype_labels_ = np.array(proto_labels)

        n = len(X)
        for step in range(self.n_iter):
            frac = step / max(self.n_iter - 1, 1)
            lr = max(self.learning_rate * np.exp(-3.0 * frac), 1e-3)
            i = int(rng.integers(n))
            point = X[i]
            dists = np.sum((prototypes - point) ** 2, axis=1)
            winner = int(np.argmin(dists))
            if self.prototype_labels_[winner] == y[i]:
                prototypes[winner] += lr * (point - prototypes[winner])
            else:
                prototypes[winner] -= lr * (point - prototypes[winner])

        self.prototypes_ = prototypes
        self.classes_ = classes
        return self

    def predict(self, X):
        """Label of the closest prototype for each row of X."""
        X = np.asarray(X, dtype=float)
        dists = np.sum(
            (X[:, None, :] - self.prototypes_[None, :, :]) ** 2, axis=2
        )
        return self.prototype_labels_[np.argmin(dists, axis=1)]


if __name__ == "__main__":
    X, labels = make_blob_data(seed=0)

    print("LVQ, 2 prototypes per class, on three blobs (seed 0)")
    model = LearningVectorQuantization(prototypes_per_class=2, n_iter=3000)
    model.fit(X, labels, seed=0)
    acc = float(np.mean(model.predict(X) == labels))
    print(f"training accuracy: {acc:.3f}")
    print("each blob has prototypes of its own, so the nearest")
    print("prototype almost always carries the right label")

    print("\nLVQ on a class split in two, other class in the middle")
    Xs, ys = make_split_class_data(seed=0)
    one = LearningVectorQuantization(prototypes_per_class=1, n_iter=3000)
    one.fit(Xs, ys, seed=0)
    acc_one = float(np.mean(one.predict(Xs) == ys))
    print(f"1 prototype per class, training accuracy: {acc_one:.3f}")
    two = LearningVectorQuantization(prototypes_per_class=2, n_iter=3000)
    two.fit(Xs, ys, seed=0)
    acc_two = float(np.mean(two.predict(Xs) == ys))
    print(f"2 prototypes per class, training accuracy: {acc_two:.3f}")
    print("one class 0 prototype can only guard one of its two")
    print("clusters, so the other cluster falls to class 1")
