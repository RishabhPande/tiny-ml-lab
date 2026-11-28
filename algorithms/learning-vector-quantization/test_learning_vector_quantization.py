"""Tests for learning vector quantization: two prototypes per
class classify three separated blobs almost perfectly, while
one prototype per class cannot cover a class that is split
into two far apart clusters with the other class in between.
"""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_learning_vector_quantization",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
LearningVectorQuantization = _impl.LearningVectorQuantization
make_blob_data = _impl.make_blob_data
make_split_class_data = _impl.make_split_class_data


def test_works():
    """On three blobs with spread 0.4 and centres about 4 units
    apart, LVQ with 2 prototypes per class reaches a training
    accuracy above 0.95, and every class ends up with at least
    one prototype within 1.0 of its blob centre, so the
    prototypes really have moved onto the clusters they guard."""
    X, labels = make_blob_data(seed=0)
    model = LearningVectorQuantization(prototypes_per_class=2, n_iter=3000)
    model.fit(X, labels, seed=0)

    accuracy = float(np.mean(model.predict(X) == labels))
    assert accuracy > 0.95

    centres = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5]])
    for cls in range(3):
        own = model.prototypes_[model.prototype_labels_ == cls]
        dists = np.sqrt(np.sum((own - centres[cls]) ** 2, axis=1))
        assert float(np.min(dists)) < 1.0


def test_breaks():
    """Class 0 lives at (-4, 0) and (4, 0) with class 1 in the
    middle at (0, 0). With 1 prototype per class, the class 0
    prototype can only settle on one of its two clusters, so
    the other cluster's points sit closer to the class 1
    prototype and accuracy stays below 0.80. With 2 prototypes
    per class, each cluster gets its own guard and accuracy
    rises above 0.95. The missing piece is a prototype, not
    more training."""
    X, labels = make_split_class_data(seed=0)

    one = LearningVectorQuantization(prototypes_per_class=1, n_iter=3000)
    one.fit(X, labels, seed=0)
    acc_one = float(np.mean(one.predict(X) == labels))
    assert acc_one < 0.80

    two = LearningVectorQuantization(prototypes_per_class=2, n_iter=3000)
    two.fit(X, labels, seed=0)
    acc_two = float(np.mean(two.predict(X) == labels))
    assert acc_two > 0.95
