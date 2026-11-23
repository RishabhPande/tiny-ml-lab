# Day 33: Non-Negative Matrix Factorization (NMF)

## Idea

NMF factors a non-negative matrix X into two non-negative
factors, W and H, so that X is close to W @ H. Nothing is ever
subtracted, so the columns of W read as parts (topics, face
pieces, spending patterns) and H says how much of each part
each sample uses. The Lee and Seung multiplicative updates move
both factors downhill on the squared reconstruction error while
keeping every entry at zero or above: each update multiplies by
a ratio of positive terms, so a factor that starts non-negative
stays non-negative.

## Run

```bash
python algorithms/nmf/impl.py
pytest algorithms/nmf/test_nmf.py
```

## Break it

`test_breaks` fits rank 2 to data built from 4 parts. The rank 4
model rebuilds the matrix to under 1 percent relative error,
while the rank 2 model stalls above 20 percent on the same data
with the same settings. A rank 2 product simply has no room for
the other two parts, so the leftover error is set by the rank
the caller chose. More iterations cannot fix it; only more
parts can, and each added part costs interpretability, which is
the trade the method makes you choose.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/nmf/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
