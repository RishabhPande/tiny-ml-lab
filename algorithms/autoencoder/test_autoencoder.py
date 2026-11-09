"""Tests for the autoencoder: a 2D bottleneck rebuilds the circle
cleanly, but a 1D bottleneck cannot hold a 2D shape, so its rebuild
error stays high no matter how long it trains."""

import os
import importlib.util

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "impl_autoencoder",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
Autoencoder = _impl.Autoencoder
make_circle_data = _impl.make_circle_data
train_autoencoder = _impl.train_autoencoder


def test_works():
    """A 2D bottleneck reconstructs the circle far better than the
    mean baseline, the latent codes actually vary, and training is
    deterministic for a fixed seed."""
    data = make_circle_data()
    baseline = float(np.mean(data ** 2))

    model, mse = train_autoencoder(data, latent_dim=2)
    assert mse < 0.05
    assert mse < 0.1 * baseline

    # The latent codes carry real information, not a constant.
    codes = model.encode(data)
    assert codes.shape == (data.shape[0], 2)
    assert np.all(codes.std(axis=0) > 0.2)

    # Rebuilds are close to the inputs, point by point.
    rebuilt = model.reconstruct(data)
    assert rebuilt.shape == data.shape
    assert np.all(np.isfinite(rebuilt))
    assert float(np.mean((data - rebuilt) ** 2)) == mse

    # Same seed, same result: training is deterministic.
    _, mse_again = train_autoencoder(data, latent_dim=2)
    assert mse_again == mse


def test_breaks():
    """A 1D bottleneck cannot hold a 2D circle: its rebuild error
    stays far above the 2D version even after the same training,
    because one number cannot describe a point on a circle. The
    failure is the bottleneck width, not the training: the 2D
    model on the same data and seed reconstructs cleanly."""
    data = make_circle_data()

    _, mse_wide = train_autoencoder(data, latent_dim=2)
    _, mse_narrow = train_autoencoder(data, latent_dim=1)
    assert mse_wide < 0.05
    assert mse_narrow > 0.15
    assert mse_narrow > 5 * mse_wide

    # Even a fresh random start does not save the narrow model.
    narrow = Autoencoder(data.shape[1], latent_dim=1, seed=7)
    mse_fresh = narrow.fit(data)
    assert mse_fresh > 0.15


def test_bad_dimensions():
    """A bottleneck narrower than one dimension is refused."""
    import pytest

    with pytest.raises(ValueError, match="latent_dim"):
        Autoencoder(8, latent_dim=0)
    with pytest.raises(ValueError, match="hidden_dim"):
        Autoencoder(8, hidden_dim=0)
