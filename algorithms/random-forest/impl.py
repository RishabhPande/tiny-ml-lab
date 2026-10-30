"""Random Forest
Bagged decision trees with random feature subsets (numpy only).
Run: python algorithms/random-forest/impl.py
"""

import numpy as np


class _Node:
    def __init__(self, counts=None, feature=None, threshold=None, left=None, right=None):
        self.counts = counts
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = counts is not None


class _Tree:
    """One CART tree, grown on one bootstrap sample with random splits."""

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng

    def _gini(self, counts):
        n = counts.sum()
        if n == 0:
            return 0.0
        p = counts / n
        return 1.0 - float(np.sum(p**2))

    def _best_split(self, X, y_idx, features):
        n = len(y_idx)
        total = np.bincount(y_idx, minlength=self.n_classes_)
        parent_gini = self._gini(total)
        best = None
        for f in features:
            order = np.argsort(X[:, f])
            Xs = X[order, f]
            ys = y_idx[order]
            diff = Xs[1:] != Xs[:-1]
            if not np.any(diff):
                continue
            one_hot = (ys[:, None] == np.arange(self.n_classes_)).astype(float)
            cum = np.cumsum(one_hot, axis=0)
            left_counts = cum[:-1][diff]
            left_n = left_counts.sum(axis=1)
            right_counts = total - left_counts
            right_n = n - left_n
            lp = left_counts / left_n[:, None]
            rp = right_counts / right_n[:, None]
            lg = 1.0 - np.sum(lp**2, axis=1)
            rg = 1.0 - np.sum(rp**2, axis=1)
            weighted = (left_n * lg + right_n * rg) / n
            i = int(np.argmin(weighted))
            gain = parent_gini - weighted[i]
            if best is None or gain > best[0]:
                pos = np.where(diff)[0][i]
                threshold = (Xs[pos] + Xs[pos + 1]) / 2.0
                best = (gain, f, threshold)
        return best

    def _grow(self, X, y_idx, depth):
        counts = np.bincount(y_idx, minlength=self.n_classes_)
        if (
            len(np.unique(y_idx)) == 1
            or len(y_idx) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return _Node(counts=counts)
        if self.max_features is None:
            features = np.arange(X.shape[1])
        else:
            m = min(self.max_features, X.shape[1])
            features = self.rng.choice(X.shape[1], size=m, replace=False)
        best = self._best_split(X, y_idx, features)
        if best is None or best[0] <= 0:
            return _Node(counts=counts)
        gain, f, threshold = best
        left_mask = X[:, f] <= threshold
        if not np.any(left_mask) or np.all(left_mask):
            return _Node(counts=counts)
        # Importance is the total impurity drop, weighted by node size.
        self.importances_[f] += gain * len(y_idx)
        left = self._grow(X[left_mask], y_idx[left_mask], depth + 1)
        right = self._grow(X[~left_mask], y_idx[~left_mask], depth + 1)
        return _Node(feature=f, threshold=threshold, left=left, right=right)

    def fit(self, X, y_idx, n_classes):
        """Grow the tree on the given (already bootstrapped) sample."""
        self.n_classes_ = n_classes
        self.importances_ = np.zeros(X.shape[1])
        self.root_ = self._grow(X, y_idx, 0)
        return self

    def _leaf_counts(self, node, x):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.counts

    def predict_proba_row(self, x):
        counts = self._leaf_counts(self.root_, x)
        return counts / counts.sum()


class RandomForestClassifier:
    """Random forest: bootstrap samples plus random feature subsets.

    Each tree sees a bootstrap resample of the rows and, at each split,
    only a random subset of the features. The forest votes, so one tree's
    quirks average out. predict returns majority-vote labels,
    predict_proba returns the mean of the trees' leaf fractions.
    """

    def __init__(self, n_estimators=50, max_depth=None, min_samples_split=2,
                 max_features="sqrt", bootstrap=True, seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.seed = seed

    def _n_features_per_split(self, n_features):
        if self.max_features is None:
            return None
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        return max(1, int(self.max_features))

    def fit(self, X, y):
        """Fit the forest. Returns self."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)
        n, d = X.shape
        m = self._n_features_per_split(d)
        self.trees_ = []
        importances = np.zeros(d)
        for i in range(self.n_estimators):
            rng = np.random.default_rng(self.seed + i)
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
            else:
                idx = np.arange(n)
            tree = _Tree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=m,
                rng=rng,
            )
            tree.fit(X[idx], y_idx[idx], self.n_classes_)
            self.trees_.append(tree)
            importances += tree.importances_
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict_proba(self, X):
        """Mean leaf fraction per class, one row per input row."""
        X = np.asarray(X, dtype=float)
        probs = np.zeros((len(X), self.n_classes_))
        for tree in self.trees_:
            for j, x in enumerate(X):
                probs[j] += tree.predict_proba_row(x)
        return probs / len(self.trees_)

    def predict(self, X):
        """Majority-vote labels."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 1.0, (150, 2))
    X1 = rng.normal(0, 1.0, (150, 2)) + np.array([3.0, 3.0])
    X = np.hstack([np.vstack([X0, X1]), rng.normal(0, 1.0, (300, 1))])
    y = np.array([0] * 150 + [1] * 150)
    perm = np.random.default_rng(42).permutation(300)
    train, test = perm[:240], perm[240:]
    model = RandomForestClassifier(n_estimators=50, seed=0).fit(X[train], y[train])
    acc = float(np.mean(model.predict(X[test]) == y[test]))
    print("held-out accuracy:", round(acc, 3))
    print("feature importances:", np.round(model.feature_importances_, 3))
    print("note: the third column is pure noise, so it should rank last")
