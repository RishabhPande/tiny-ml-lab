# Day 28: Spectral Clustering

## Idea

Spectral clustering clusters a graph, not the raw points.
Each pair of points gets a similarity weight from a Gaussian
kernel with width sigma, the weights form the graph Laplacian,
and the smallest eigenvectors of that Laplacian give a new
space where strongly connected points sit close together.
K-means in that space does the actual split. The payoff shows
on two concentric rings: both rings share one center, so no
center based method can separate them, yet the graph sees the
inner ring as one connected loop and the outer ring as
another, and the demo recovers both rings exactly.

## Run

```bash
python algorithms/spectral-clustering/impl.py
pytest algorithms/spectral-clustering/test_spectral-clustering.py
```

## Break it

`test_breaks` keeps two plain blobs, 10 spreads apart, that
come back exactly at sigma 0.5, then builds their graph with
sigma 0.0001. At that extreme width every off-diagonal Gaussian
weight underflows to zero. The test checks that no edges
survive, for both original and shuffled rows, and that every
normalized-Laplacian eigenvalue equals 1.

This is a degenerate eigenspace, not a promise of a specific
bad accuracy. An eigensolver may return any basis for that
space; row order and numerical details can accidentally give
labels that agree with the blobs. The demo prints the observed
accuracy but uses the erased graph to explain the failure.
The blobs did not move; the bandwidth removed the relationships
that spectral clustering needs.

Sigma too large fails the other way, blurring separate groups
into one connected mass. Try the demo's ring data at sigma 1.0.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/spectral-clustering/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
