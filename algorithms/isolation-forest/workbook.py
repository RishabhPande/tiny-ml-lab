"""Isolation Forest workbook: fill in the TODO, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import IsolationForest, expected_path_length, make_data  # noqa: E402


class WorkbookForest(IsolationForest):
    """TODO: an isolation forest that turns path lengths into scores.

    The trees and the path lengths are given. Fill in
    `anomaly_score` so a short average path becomes a score
    near 1 and an ordinary path becomes a score near 0.5.
    """

    def anomaly_score(self, x):
        """TODO: rescale the average path length into a score.

        score = 2 ** (-average_path / c(psi)), where psi is
        self.psi, c is expected_path_length, and average_path is
        self.average_path_length(x). Hint: a point whose average
        path equals c(psi) should score exactly 0.5.
        """
        raise NotImplementedError("TODO: implement anomaly_score")


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


if __name__ == "__main__":
    x, y = make_data()
    forest = WorkbookForest().fit(x)
    results = [
        _check("anomaly_score", lambda: forest.anomaly_score(x)),
    ]
    if all(results):
        scores = forest.anomaly_score(x)
        print(f"Mean score, normal points: {scores[y == 0].mean():.3f}")
        print(f"Mean score, outliers: {scores[y == 1].mean():.3f}")
        print("All parts implemented. Now run: "
              "pytest test_isolation-forest.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
