# Day 14: Gaussian Mixture Model (GMM)

## Idea

A Gaussian mixture says the data came from several Gaussian blobs
mixed together, and each point belongs to each blob with some
probability. Expectation-maximization (EM) fits that story: the E
step works out, for every point, how much each blob is responsible
for it, and the M step refits each blob's mean, covariance, and
weight to the points it is responsible for. Because every component
gets its own full covariance, clusters can be stretched and rotated,
not just round. A small ridge on each covariance keeps a component
from collapsing onto a single point.

## Run

```bash
python algorithms/gmm/impl.py
pytest algorithms/gmm/test_gmm.py
```

## Break it

`test_breaks` fits two components to blobs whose centers sit only
0.5 apart with unit spread, so the blobs overlap almost completely.
Even the true model cannot tell most points apart, and the fitted
mixture scores well below its separated-blob accuracy while its
average confidence drops toward 0.5. The optimizer is fine, the
model class is the limit: a mixture of Gaussians can only separate
what the Gaussians themselves keep apart. That is the cue to check
overlap before trusting hard cluster labels, and to read the soft
responsibilities instead of just the winning component.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/gmm/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
