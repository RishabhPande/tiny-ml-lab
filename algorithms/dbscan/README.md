# Day 10: DBSCAN

## Idea

DBSCAN groups points by density instead of by distance to a center. A
point is a core point when at least `min_samples` points lie within
`eps` of it. Core points linked by overlapping neighborhoods join one
cluster, a non-core point near a core point joins as a border point,
and everything else is noise, labeled -1. That gives two things
K-Means cannot: clusters of any shape, and an explicit noise label
instead of forcing every point into a cluster.

## Run

```bash
python algorithms/dbscan/impl.py
pytest algorithms/dbscan/test_dbscan.py
```

## Break it

`test_breaks` pairs a dense blob (spread 0.2) with a sparse blob
(spread 1.2) under one radius, eps 0.5. The dense blob is found
cleanly, but most of the sparse blob is labeled noise, even though
it is a real group. One radius cannot serve two densities at once:
raise eps to rescue the sparse blob and you risk merging separate
groups or swallowing noise. When densities vary, scale the features,
tune eps per region, or reach for a hierarchical density method
instead of trusting a single radius.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/dbscan/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
