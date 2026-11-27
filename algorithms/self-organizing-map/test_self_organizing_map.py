"""Tests for the self-organizing map: an 8 by 8 grid trained
on three blobs represents them with a small quantization
error and gives each blob its own busiest unit, while a 1 by 1
grid collapses all three blobs onto a single prototype and
leaves every point far from it.
"""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_self_organizing_map",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
SelfOrganizingMap = _impl.SelfOrganizingMap
make_blob_data = _impl.make_blob_data


def test_works():
    """On three blobs with spread 0.4 and centres about 4 units
    apart, a trained 8 by 8 map reaches a quantization error
    below 0.35, and the busiest unit of each blob is a different
    unit, at least 2 grid steps from the other blobs' busiest
    units, so the grid really has separated the clusters."""
    X, labels = make_blob_data(seed=0)
    som = SelfOrganizingMap(grid_rows=8, grid_cols=8, n_iter=4000)
    som.fit(X, seed=0)

    assert som.quantization_error(X) < 0.35

    units = som.bmu(X)
    modes = []
    for blob in range(3):
        cells = units[labels == blob]
        keys = cells[:, 0] * 8 + cells[:, 1]
        values, counts = np.unique(keys, return_counts=True)
        modes.append(int(values[np.argmax(counts)]))
    assert len(set(modes)) == 3
    cells = [(m // 8, m % 8) for m in modes]
    for i in range(3):
        for j in range(i + 1, 3):
            dist = abs(cells[i][0] - cells[j][0]) + abs(cells[i][1] - cells[j][1])
            assert dist >= 2


def test_breaks():
    """A 1 by 1 grid has a single prototype, so on the same
    three-blob data its quantization error stays above 1.5
    where the 8 by 8 map scores below 0.35, and every point is
    assigned to the same one unit. The blobs are far apart and
    no training schedule can fix that: the capacity is missing,
    not the optimisation. Only a bigger grid helps."""
    X, labels = make_blob_data(seed=0)

    tiny = SelfOrganizingMap(grid_rows=1, grid_cols=1, n_iter=4000)
    tiny.fit(X, seed=0)
    assert tiny.quantization_error(X) > 1.5
    assert len(np.unique(tiny.bmu(X), axis=0)) == 1

    big = SelfOrganizingMap(grid_rows=8, grid_cols=8, n_iter=4000)
    big.fit(X, seed=0)
    assert big.quantization_error(X) < 0.35
