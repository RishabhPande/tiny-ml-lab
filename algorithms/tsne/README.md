# Day 29: t-SNE

## Idea

t-SNE builds a map, not a model. Each point gets a probability
distribution over its neighbors, with a Gaussian kernel whose
width is tuned point by point so every distribution has the
same effective neighbor count, the perplexity. Points are then
placed in 2D so a matching distribution, this time with a heavy
tailed Student t kernel, fits the first one as well as gradient
descent can manage. Neighbors stay neighbors, so clusters that
show up on the map are genuinely close in the original space,
and the demo keeps a perfect nearest neighbor vote when three
8 dimensional blobs are flattened to a picture.

## Run

```bash
python algorithms/tsne/impl.py
pytest algorithms/tsne/test_tsne.py
```

## Break it

`test_breaks` uses three blobs in a line: A and B are 3 units
apart, C is 30 units from A, so the true far to near centroid
distance ratio is near 10. On the map that ratio falls under
2. t-SNE never promised global geometry, only local
neighborhoods, so it spreads the clusters near evenly and the
spacing information is simply discarded. Cluster sizes on the
map are invented the same way. A t-SNE plot can tell you who
sits next to whom; it cannot tell you how far apart groups are
or how spread out they really were, and treating the picture
as a scale drawing is the standard way this method misleads.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/tsne/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
