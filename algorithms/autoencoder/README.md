# Day 19: Autoencoder (MLP Bottleneck)

## Idea

An autoencoder learns to squeeze its input through a narrow
bottleneck and then rebuild it on the other side. The encoder
maps the input down to a few latent numbers, the decoder maps
those numbers back up, and training minimizes the rebuild error.
This day trains a small numpy MLP autoencoder, with plain
backprop and Adam, on 400 points from a noisy circle embedded
in 8 dimensions. A circle is a 2D shape, so a 2D bottleneck can
hold it, but a 1D bottleneck cannot.

## Run

```bash
python algorithms/autoencoder/impl.py
pytest algorithms/autoencoder/test_autoencoder.py
```

## Break it

`test_breaks` trains the same autoencoder with a 1D bottleneck
instead of 2D. Its rebuild error stays above 0.15, more than
five times the 2D version's 0.03, even after the same training
and from a fresh random start. One number cannot describe a
point on a circle, so the shape is lost at the bottleneck and
no amount of training gets it back. That is the whole point of
the bottleneck: its width is a hard limit on what the model
can remember, which is also why autoencoders double as anomaly
detectors, since inputs off the learned shape rebuild badly.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/autoencoder/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
