# Day 2: K-Means

Lloyd's algorithm: pick centers, assign each point to the nearest center,
move each center to its cluster mean, repeat until the centers stop moving.
The k-means++ init spreads the starting centers out so the result is less
of a lottery.

## Run

```bash
python algorithms/kmeans/impl.py
pytest algorithms/kmeans/test_kmeans.py
```

## Do it yourself

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/kmeans/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.

## What the break-it test shows

`test_breaks` builds two concentric rings. K-means only understands round,
similar-sized clusters, so it slices both rings by angle instead of
separating inner from outer. When your clusters are not blobs, you need a
method that sees shape or density, not just distance to a center.
