# Day 34: Mean Shift

## Idea

Mean shift clusters without being told how many clusters to
find. A walk starts from every point: take the kernel weighted
centre of the points inside a window set by the bandwidth,
move there, and repeat until the walk stops. Points whose
walks end at the same peak form a cluster, so the peaks set
the cluster count. A Gaussian kernel weights nearby points
more, which keeps the walks smooth.

## Run

```bash
python algorithms/mean-shift/impl.py
pytest algorithms/mean-shift/test_mean_shift.py
```

## Break it

`test_breaks` runs bandwidth 8.0 on two blobs 6 units apart.
The window covers both blobs at once, so the weighted centre
always sits between them and every walk climbs to one shared
peak: two clusters come back as one. Bandwidth 2.0 on the same
data finds both. The bandwidth is the whole method: too wide
merges real clusters, too narrow turns noise bumps into
clusters of their own, and no step count fixes a wrong window.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/mean-shift/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
