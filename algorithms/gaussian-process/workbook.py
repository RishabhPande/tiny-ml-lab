"""Gaussian process workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_wave, r2_score, rbf_kernel  # noqa: E402


class GaussianProcessRegressor:
    """TODO: Gaussian process regression with an RBF kernel."""

    def __init__(self, length_scale=1.0, signal_std=1.0, noise_std=0.1):
        if length_scale <= 0:
            raise ValueError("length_scale must be positive")
        if signal_std <= 0:
            raise ValueError("signal_std must be positive")
        if noise_std < 0:
            raise ValueError("noise_std must not be negative")
        self.length_scale = length_scale
        self.signal_std = signal_std
        self.noise_std = noise_std

    def fit(self, X, y):
        """TODO: store X and y, build the kernel matrix with
        rbf_kernel, add noise_std squared plus a small jitter to
        its diagonal, Cholesky factorise it into self.L_, and
        solve for self.alpha_ against y. Return self."""
        raise NotImplementedError("TODO: implement fit")

    def predict(self, X):
        """TODO: build the kernel between the training inputs
        and X, return the posterior mean (kernel transposed
        times self.alpha_) and the posterior standard deviation
        (signal variance minus the solved quadratic form, square
        rooted, clipped at zero)."""
        raise NotImplementedError("TODO: implement predict")


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
    X_train, y_train, X_test, y_test = make_wave(seed=0)
    model = GaussianProcessRegressor(length_scale=1.0, noise_std=0.1)
    results = [
        _check("fit", lambda: model.fit(X_train, y_train)),
        _check("predict", lambda: model.predict(X_test)),
    ]
    if all(results):
        mean, std = model.predict(X_test)
        print(f"test R2: {r2_score(y_test, mean):.3f}")
        print("All parts implemented. Now run: pytest test_gaussian_process.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
