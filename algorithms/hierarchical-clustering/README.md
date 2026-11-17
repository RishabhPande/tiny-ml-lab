# Day 27: Hierarchical Clustering

## Idea

Agglomerative clustering starts with every point in its own
cluster and merges the two closest clusters, one step at a
time, until a stopping rule says enough. Closest is defined
by the linkage: single linkage uses the nearest pair of points
across two clusters, complete linkage uses the farthest pair,
and average linkage uses the mean of all cross pairs. The
merge history forms a dendrogram, so one run can be cut at
different heights to give different cluster counts. On the
demo here, three blobs about 15 spreads apart come back
exactly, under all three linkages: every found cluster holds
points from only one true blob.

## Run

```bash
python algorithms/hierarchical-clustering/impl.py
pytest algorithms/hierarchical-clustering/test_hierarchical-clustering.py
```

## Break it

`test_breaks` uses single linkage with a distance threshold
of 1.0. Two blobs 20 spreads apart, with nothing between
them, give exactly two clusters. Then 13 bridge points are
added between the blobs, spaced about 0.4 apart, and the same
settings give one cluster. Single linkage only ever looks at
the nearest pair, so each small step along the chain counts
as contact, and the merges walk across a gap of 6 that no
single step could cross. The blobs did not move; a thin
trail of points was enough to chain them together. This is
the chaining effect, and it is why single linkage struggles
on data with stray points strung between real groups.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/hierarchical-clustering/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
