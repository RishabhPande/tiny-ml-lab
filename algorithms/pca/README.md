# Day 6: PCA

## Idea

Center the data, eigendecompose its covariance, and keep the axes with the
largest eigenvalues. Those axes are the directions where the data varies
most, and projecting onto them is the best linear compression in the
least-squares sense.

## Run

```bash
python algorithms/pca/impl.py
pytest algorithms/pca/test_pca.py
```

## Break it

`test_breaks` puts points around a circle. A circle has no single best
line: every direction varies about equally, so the first component holds
only half the variance. Linear PCA cannot represent curved structure, no
matter how many points you give it. That gap is what kernels and
autoencoders are for.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/pca/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
