"""PageRank workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def adjacency_from_edges(n_pages, edges):
    """TODO: build adj where adj[i, j] is 1 when page i links to j."""
    raise NotImplementedError("TODO: implement adjacency_from_edges")


class PageRank:
    """TODO: random-surfer ranking by power iteration."""

    def __init__(self, damping=0.85, max_iter=100, tol=1e-6):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, adj):
        """TODO: rank the pages. Set ranks_ (sums to 1) and n_iter_."""
        raise NotImplementedError("TODO: implement fit")


def _check(name, fn):
    try:
        fn()
    except NotImplementedError:
        print(f"[TODO] {name} is not implemented yet")
        return False
    except Exception as exc:  # noqa: BLE001 - workbook feedback, not a test
        print(f"[ERROR] {name} raised {type(exc).__name__}: {exc}")
        return False
    print(f"[OK] {name} runs")
    return True


def _ranks_sum_to_one(model):
    if abs(model.ranks_.sum() - 1.0) >= 1e-9:
        raise AssertionError("ranks do not sum to 1")


if __name__ == "__main__":
    holder = {}
    model = PageRank()

    def _build():
        holder["adj"] = adjacency_from_edges(3, [(0, 1), (1, 2), (2, 0)])

    def _fit():
        model.fit(holder["adj"])

    results = [
        _check("adjacency_from_edges", _build),
        _check("fit", _fit),
        _check("ranks sum to one", lambda: _ranks_sum_to_one(model)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_pagerank.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
