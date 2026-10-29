"""Decision Tree
CART-style decision tree with Gini splits (numpy only).
Run: python algorithms/decision-tree/impl.py
"""

import numpy as np


class _Node:
    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = value is not None


class DecisionTree:
    """Axis-aligned tree. max_depth=None grows until leaves are pure."""

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _gini(self, counts):
        n = counts.sum()
        if n == 0:
            return 0.0
        p = counts / n
        return 1.0 - float(np.sum(p**2))

    def _best_split(self, X, y_idx):
        n = len(y_idx)
        total = np.bincount(y_idx, minlength=self.n_classes_)
        parent_gini = self._gini(total)
        best = None
        for f in range(X.shape[1]):
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
        value = self.classes_[int(np.argmax(counts))]
        if (
            len(np.unique(y_idx)) == 1
            or len(y_idx) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return _Node(value=value)
        best = self._best_split(X, y_idx)
        if best is None or best[0] <= 0:
            return _Node(value=value)
        _, f, threshold = best
        left_mask = X[:, f] <= threshold
        if not np.any(left_mask) or np.all(left_mask):
            return _Node(value=value)
        left = self._grow(X[left_mask], y_idx[left_mask], depth + 1)
        right = self._grow(X[~left_mask], y_idx[~left_mask], depth + 1)
        return _Node(feature=f, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        """Grow the tree. Returns self."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)
        self.tree_ = self._grow(X, y_idx, 0)
        return self

    def _predict_one(self, x, node):
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        """Predict one label per row of X."""
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def depth(self):
        """Depth of the tree (root counts as 0)."""

        def _depth(node):
            if node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        return _depth(self.tree_)


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_train = np.vstack(
        [rng.normal(0, 1, (100, 2)), rng.normal(0, 1, (100, 2)) + [4, 0]]
    )
    y_train = np.array([0] * 100 + [1] * 100)
    tree = DecisionTree(max_depth=4).fit(X_train, y_train)
    print("train accuracy:", round(accuracy(y_train, tree.predict(X_train)), 3))
    print("tree depth:", tree.depth())
