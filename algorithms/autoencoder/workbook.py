"""Autoencoder workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_circle_data, _tanh, _tanh_grad  # noqa: E402


class Autoencoder:
    """TODO: a small MLP autoencoder trained with Adam and backprop.

    Architecture: input -> hidden (tanh) -> latent (linear) ->
    hidden (tanh) -> output (linear). The latent layer is the
    bottleneck: its width decides how much of the input can
    survive the round trip.
    """

    def __init__(self, input_dim, latent_dim=2, hidden_dim=16,
                 seed=0):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(1.0 / input_dim)
        scale2 = np.sqrt(1.0 / hidden_dim)
        scale3 = np.sqrt(1.0 / latent_dim)
        self.params = {
            "W1": rng.standard_normal((input_dim, hidden_dim)) * scale1,
            "b1": np.zeros(hidden_dim),
            "W2": rng.standard_normal((hidden_dim, latent_dim)) * scale2,
            "b2": np.zeros(latent_dim),
            "W3": rng.standard_normal((latent_dim, hidden_dim)) * scale3,
            "b3": np.zeros(hidden_dim),
            "W4": rng.standard_normal((hidden_dim, input_dim)) * scale2,
            "b4": np.zeros(input_dim),
        }

    def _forward(self, x):
        p = self.params
        h1 = _tanh(x @ p["W1"] + p["b1"])
        z = h1 @ p["W2"] + p["b2"]
        h2 = _tanh(z @ p["W3"] + p["b3"])
        out = h2 @ p["W4"] + p["b4"]
        cache = (x, h1, z, h2, out)
        return out, cache

    def _backward(self, cache):
        """TODO: gradients of the mean squared rebuild error.

        The loss is mean((out - x) ** 2) over every element. Work
        backwards from the output: d_out first, then through W4,
        the tanh of h2, W3, W2, and the tanh of h1. Return a dict
        with the same keys as self.params. Hint: the derivative of
        tanh is 1 - tanh(a) ** 2, where a is the pre-activation.
        """
        raise NotImplementedError("TODO: implement _backward")

    def fit(self, x, epochs=300, lr=0.01, seed=0):
        """Train by full-batch Adam on mean squared rebuild error."""
        del seed  # weights are seeded in __init__; training is fixed
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for step in range(1, epochs + 1):
            out, cache = self._forward(x)
            grads = self._backward(cache)
            for k in self.params:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                m_hat = m[k] / (1 - beta1 ** step)
                v_hat = v[k] / (1 - beta2 ** step)
                self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return float(self.reconstruction_error(x))

    def encode(self, x):
        """Map inputs to their latent codes."""
        p = self.params
        h1 = _tanh(x @ p["W1"] + p["b1"])
        return h1 @ p["W2"] + p["b2"]

    def reconstruct(self, x):
        """Rebuild inputs from their latent codes."""
        out, _ = self._forward(x)
        return out

    def reconstruction_error(self, x):
        """Mean squared error between inputs and their rebuilds."""
        return float(np.mean((x - self.reconstruct(x)) ** 2))


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
    data = make_circle_data()
    model = Autoencoder(data.shape[1], latent_dim=2)
    results = [
        _check("backward",
               lambda: model._backward(model._forward(data)[1])),
        _check("fit", lambda: model.fit(data)),
    ]
    if all(results):
        print(f"Rebuild MSE: {model.reconstruction_error(data):.4f}")
        print("All parts implemented. Now run: pytest test_autoencoder.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
