"""Autoencoder: a tiny MLP that compresses data and rebuilds it.

An autoencoder learns to squeeze its input through a narrow
bottleneck and then rebuild it on the other side. The encoder
maps the input down to a few latent numbers, the decoder maps
those numbers back up, and training minimizes the rebuild error.
If the bottleneck is wide enough to hold the data's real shape,
the rebuild is close. If it is too narrow, information is lost
for good, and no amount of training gets it back.

This day trains a small numpy MLP autoencoder on points from a
noisy circle embedded in 8 dimensions. A circle is a 2D shape,
so a 2D bottleneck can hold it, but a 1D bottleneck cannot: the
break-it test shows the 1D version stuck at a high rebuild error
while the 2D version reconstructs cleanly.

Run: python algorithms/autoencoder/impl.py
"""

import numpy as np


def make_circle_data(n=400, dim=8, noise=0.05, seed=0):
    """Points on a noisy circle, embedded in higher dimensions.

    Each point starts as (cos t, sin t) on the unit circle, then a
    fixed random map spreads it into `dim` dimensions and a little
    Gaussian noise is added. Columns are standardized so every
    feature has zero mean and unit variance.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    base = np.column_stack([np.cos(theta), np.sin(theta)])
    embed = rng.standard_normal((2, dim)) / np.sqrt(dim)
    x = base @ embed + noise * rng.standard_normal((n, dim))
    return (x - x.mean(axis=0)) / x.std(axis=0)


def _tanh(x):
    return np.tanh(x)


def _tanh_grad(x):
    return 1.0 - np.tanh(x) ** 2


class Autoencoder:
    """Small MLP autoencoder trained with Adam and plain backprop.

    Architecture: input -> hidden (tanh) -> latent (linear) ->
    hidden (tanh) -> output (linear). The latent layer is the
    bottleneck: its width decides how much of the input can
    survive the round trip.
    """

    def __init__(self, input_dim, latent_dim=2, hidden_dim=16,
                 seed=0):
        if latent_dim < 1:
            raise ValueError("latent_dim must be at least 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")
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
        x, h1, z, h2, out = cache
        p = self.params
        n = x.shape[0]
        d_out = 2.0 * (out - x) / (x.size)
        grads = {}
        grads["W4"] = h2.T @ d_out
        grads["b4"] = d_out.sum(axis=0)
        d_h2 = (d_out @ p["W4"].T) * _tanh_grad(z @ p["W3"] + p["b3"])
        grads["W3"] = z.T @ d_h2
        grads["b3"] = d_h2.sum(axis=0)
        d_z = d_h2 @ p["W3"].T
        grads["W2"] = h1.T @ d_z
        grads["b2"] = d_z.sum(axis=0)
        d_h1 = (d_z @ p["W2"].T) * _tanh_grad(x @ p["W1"] + p["b1"])
        grads["W1"] = x.T @ d_h1
        grads["b1"] = d_h1.sum(axis=0)
        return grads

    def fit(self, x, epochs=300, lr=0.01, seed=0):
        """Train by full-batch Adam on mean squared rebuild error.

        Returns the final mean squared error. Deterministic for a
        fixed seed: the weights start from it and Adam has no
        randomness of its own.
        """
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


def train_autoencoder(x, latent_dim=2, epochs=300, seed=0):
    """Build and train an autoencoder, returning (model, final mse)."""
    model = Autoencoder(x.shape[1], latent_dim=latent_dim, seed=seed)
    mse = model.fit(x, epochs=epochs, seed=seed)
    return model, mse


if __name__ == "__main__":
    data = make_circle_data()
    print(f"Autoencoder on {data.shape[0]} circle points "
          f"in {data.shape[1]} dims (fixed seed)")
    print(f"Baseline MSE (predict the mean, all zeros): "
          f"{float(np.mean(data ** 2)):.4f}")

    # The works case: a 2D bottleneck can hold a 2D circle.
    model, mse = train_autoencoder(data, latent_dim=2)
    print(f"\nlatent_dim=2: final rebuild MSE {mse:.4f}")
    print(f"  latent codes span: "
          f"{model.encode(data).std(axis=0).round(2)}")

    # The break case: a 1D bottleneck cannot hold a 2D circle.
    small, mse_small = train_autoencoder(data, latent_dim=1)
    print(f"\nlatent_dim=1: final rebuild MSE {mse_small:.4f}")
    print("  One number cannot describe a point on a circle: the")
    print("  rebuild stays blurry no matter how long it trains,")
    print("  because the bottleneck throws the shape away.")
