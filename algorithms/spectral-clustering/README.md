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
come back exactly at sigma 0.5, then reruns with sigma 0.01.
At that width a pair of points 0.1 apart already has kernel
weight near zero, so almost no edge survives and the graph
falls apart into near isolated points. The Laplacian
eigenvectors stop describing the blobs, and accuracy drops
under 0.75, close to a coin flip. The blobs did not move;
sigma quietly decided what counts as connected, and with the
graph gone the method has nothing left to cluster. Sigma too
large fails the other way, blurring separate groups into one
connected mass, which the demo's ring data shows at sigma 1.0
if you try it.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/spectral-clustering/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
