"""Tests for Isolation Forest: lone outliers far from a normal
blob are isolated in a few random splits and score near the top,
but a tight cluster of anomalies masks itself, because each
member still needs several splits to be separated from its
neighbours, so the cluster scores as ordinary as normal data.
"""

import os
import importlib.util

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "impl_isolation_forest",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
IsolationForest = _impl.IsolationForest
make_data = _impl.make_data
make_clustered_data = _impl.make_clustered_data
recall_at_k = _impl.recall_at_k


def test_works():
    """The 5 lone outliers are the 5 highest scores, well clear
    of the normal points, and a fixed seed gives the same scores
    every time."""
    x, y = make_data()
    forest = IsolationForest().fit(x)
    scores = forest.anomaly_score(x)

    # Every outlier outranks every normal point.
    assert scores[y == 1].min() > scores[y == 0].max()
    assert recall_at_k(scores, y, 5) == 1.0

    # The gap is large, not a coin flip at the boundary.
    assert scores[y == 1].mean() > 0.65
    assert scores[y == 0].mean() < 0.50
    assert scores[y == 1].mean() - scores[y == 0].mean() > 0.2

    # Scores are a real rescaling of path lengths: lone outliers
    # really do take shorter paths through the trees.
    paths = forest.average_path_length(x)
    assert paths[y == 1].max() < paths[y == 0].mean()

    # Same seed, same forest: scoring is deterministic.
    scores_again = IsolationForest().fit(x).anomaly_score(x)
    assert np.array_equal(scores, scores_again)


def test_breaks():
    """Masking: 100 anomalies in one tight cluster at (5, 5)
    score as ordinary as the normal points and most are missed,
    while 100 anomalies scattered alone at the same distance
    are caught almost perfectly. The failure is the clustering,
    not the distance or the forest."""
    x_scattered, x_clustered = make_clustered_data()
    y = np.array([0] * 150 + [1] * 100)

    scattered = IsolationForest().fit(x_scattered)
    scattered_scores = scattered.anomaly_score(x_scattered)
    clustered = IsolationForest().fit(x_clustered)
    clustered_scores = clustered.anomaly_score(x_clustered)

    # Scattered lone anomalies are caught almost perfectly.
    assert recall_at_k(scattered_scores, y, 100) > 0.9
    assert scattered_scores[y == 1].mean() > 0.55

    # The tight cluster hides: most members are never flagged,
    # and on average they score no higher than normal points.
    assert recall_at_k(clustered_scores, y, 100) < 0.5
    assert clustered_scores[y == 1].mean() < 0.52
    assert (clustered_scores[y == 1].mean()
            < scattered_scores[y == 1].mean() - 0.05)


def test_constant_features_do_not_stop_splittable_trees():
    """Irrelevant constant columns must not turn useful trees into leaves."""
    x, y = make_data()
    padded = np.column_stack([np.zeros((len(x), 50)), x])
    forest = IsolationForest().fit(padded)

    # Every sampled root has variable coordinates, even if the first
    # randomly chosen column is constant.
    assert all(tree.feature is not None for tree in forest.trees)
    scores = forest.anomaly_score(padded)
    assert np.isfinite(scores).all()
    assert recall_at_k(scores, y, 5) == 1.0
    again = IsolationForest().fit(padded).anomaly_score(padded)
    assert np.array_equal(scores, again)


def test_all_constant_features_remain_a_leaf():
    """A genuinely unsplittable node terminates with its size correction."""
    x = np.ones((16, 3))
    forest = IsolationForest(n_trees=5, sample_size=16).fit(x)
    assert all(tree.feature is None and tree.size == 16
               for tree in forest.trees)
    assert np.array_equal(forest.anomaly_score(x), np.full(16, 0.5))
