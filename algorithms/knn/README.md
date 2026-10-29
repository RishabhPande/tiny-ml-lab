# Day 3: K-Nearest Neighbors

No training, just memory: to classify a new point, look at the k closest
training points and take a majority vote. Ties go to the smallest label.

## Run

```bash
python algorithms/knn/impl.py
pytest algorithms/knn/test_knn.py
```

## Do it yourself

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/knn/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.

## What the break-it test shows

`test_breaks` keeps the same two informative features and adds 50 noisy
ones. Distances get swamped by noise, so "nearest" stops meaning "most
similar" and accuracy collapses. That is the curse of dimensionality in
one test: kNN needs feature selection or scaling the moment most features
are irrelevant.
