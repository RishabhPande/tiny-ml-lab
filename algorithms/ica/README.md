# Day 32: Independent Component Analysis (ICA)

## Idea

ICA unmixes linearly mixed signals. The data are centered and
whitened, which decorrelates them, and then FastICA rotates the
whitened data until each component is as non-Gaussian as possible,
using a tanh contrast in a fixed-point iteration with symmetric
decorrelation. Decorrelation alone (what PCA does) leaves every
rotation open; non-Gaussianity is the extra signal that picks one
rotation out, the one whose components look like independent
sources. The recovered components come back in no promised order,
with arbitrary sign and scale, because the mixtures carry no
information about any of those.

## Run

```bash
python algorithms/ica/impl.py
pytest algorithms/ica/test_ica.py
```

## Break it

`test_breaks` mixes three independent Gaussian sources with the
same mixing matrix that the works test unmixes almost perfectly.
A rotation of independent Gaussians is still a set of independent
Gaussians, so every candidate rotation scores exactly the same
and there is nothing for the iteration to aim at: the matched
correlation with the true sources lands around 0.77 instead of
above 0.99. ICA does not fail on Gaussian sources because it is
weak; the sources are mathematically unidentifiable from the
mixtures, and no tuning changes that.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/ica/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
