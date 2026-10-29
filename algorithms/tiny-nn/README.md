# Day 7: Tiny Neural Net

One hidden layer, tanh activations, sigmoid output, binary cross-entropy,
full-batch gradient descent. It is the smallest network that is still a
real network: no linear model can solve XOR, and this one can.

## Run

```bash
python algorithms/tiny-nn/impl.py
pytest algorithms/tiny-nn/test_tiny-nn.py
```

## Do it yourself

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/tiny-nn/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.

## What the break-it test shows

`test_breaks` gives the network two features: one that separates the
classes, and one that is uniform noise from 0 to 1000. Without scaling,
the noise feature saturates tanh, gradients vanish, and the network learns
nothing. With standardization, the same network learns fine. Scaling is not
a preprocessing nicety: for saturating activations, it is the difference
between learning and not learning.
