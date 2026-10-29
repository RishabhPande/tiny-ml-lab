# Day 4: Gaussian Naive Bayes

## Idea

Assume every feature is an independent normal per class, multiply the
evidence, and pick the class with the highest posterior. The "naive" part
is the independence assumption, and it is also the whole trick: it makes
the model tiny, fast, and surprisingly hard to beat on text-like data.

## Run

```bash
python algorithms/naive-bayes/impl.py
pytest algorithms/naive-bayes/test_naive-bayes.py
```

## Break it

`test_breaks` takes a weak feature and duplicates it 100 times. Naive Bayes
counts each copy as fresh evidence, so the weak feature outvotes the strong
one and flips the prediction. Correlated features break the independence
assumption, and the model does not know it is double-counting.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/naive-bayes/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
