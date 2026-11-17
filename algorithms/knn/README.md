# Day 3: K-Nearest Neighbors

## Idea

No training, just memory: to classify a new point, look at the k closest
training points and take a majority vote. Vote ties go to the smallest label.
`n_neighbors` must be an integer from 1 through the number of training rows,
including all rows. Invalid values raise `ValueError` at prediction time.
Equal-distance points at the neighbor cutoff have no guaranteed selection order.

## Run

```bash
python algorithms/knn/impl.py
pytest algorithms/knn/test_knn.py
```

## Break it

`test_breaks` keeps the same two informative features and adds 50 noisy
ones. Distances get swamped by noise, so "nearest" stops meaning "most
similar" and accuracy collapses. That is the curse of dimensionality in
one test: kNN needs feature selection or scaling the moment most features
are irrelevant.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/knn/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
