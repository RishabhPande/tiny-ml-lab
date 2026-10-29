# Day 1: Linear Regression

## Idea

Closed-form least squares: solve for the line (or hyperplane) that
minimizes squared error, in one shot, with no iteration.

## Run

```bash
python algorithms/linear-regression/impl.py
pytest algorithms/linear-regression/test_linear-regression.py
```

## Break it

`test_breaks` adds a single huge outlier at the edge of the x range. Least
squares trusts every point equally, so one bad point drags the whole line.
That is why robust losses and outlier handling exist: the method is optimal
for clean Gaussian noise and fragile the moment that assumption breaks.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/linear-regression/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
