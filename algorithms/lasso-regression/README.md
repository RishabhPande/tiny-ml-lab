# Day 25: Lasso Regression

## Idea

The lasso (Tibshirani, 1996) is least squares with a penalty on
the sum of the absolute values of the coefficients, fitted here
by coordinate descent with a soft-threshold and the intercept
left unpenalized. Switching the penalty from squared size (Day
24, ridge) to absolute size sounds minor and is not: the
soft-threshold can hold a coefficient at exactly zero, so the
fit doubles as feature selection. On the sparse demo here, ten
features go in and only three matter, and lasso at alpha 0.1
sets all seven useless coefficients to exactly 0 while scoring
held-out R2 of 0.978. Least squares on the same data gives
every useless feature a small nonzero weight instead, the
smallest of them 0.013, because nothing in its objective can
produce an exact zero.

## Run

```bash
python algorithms/lasso-regression/impl.py
pytest algorithms/lasso-regression/test_lasso-regression.py
```

## Break it

`test_breaks` builds two features that are near copies of one
hidden factor and splits the true effect evenly between them,
(2.0, 2.0). The lasso hands almost all of it to one twin,
(3.924, 0.0): the second feature is deleted outright, as if it
carried nothing. Held-out R2 stays at 0.984 and the combined
effect is about right, so the predictions give no warning. The
failure only shows when the coefficients are read as an
explanation of which features matter, which is exactly what
people reach for a sparse model to do. With correlated
features, which twin survives is an accident of the data and
the sweep order, not a finding. Day 24 shows the opposite
temperament on the same kind of data: ridge shares the effect
across the copies instead of picking one.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/lasso-regression/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
