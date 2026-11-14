# Day 24: Ridge Regression

## Idea

Ridge regression (Hoerl and Kennard, 1970) is least squares
with a penalty on the size of the coefficients. Centering the
data first so the intercept stays unpenalized, it solves
(X^T X + alpha I) w = X^T y instead of the plain normal
equations. The penalty buys stability: when features are near
copies of each other, least squares splits the effect between
them with huge, opposing coefficients that cancel in training
and wobble on new data, while ridge keeps the coefficients
small and shared. On the collinear demo here, ridge at
alpha 10 has a coefficient norm of 1.128 against 3.212 for
least squares, and gives up almost nothing: held-out R2 of
0.927 against 0.932.

## Run

```bash
python algorithms/ridge-regression/impl.py
pytest algorithms/ridge-regression/test_ridge-regression.py
```

## Break it

`test_breaks` turns alpha up to 1e6 on clean data. At that
point the penalty, not the data, decides the fit: every
coefficient shrinks to under 0.001 in size, the model predicts
close to the mean everywhere, and held-out R2 falls to -0.048,
worse than just guessing the average, against 0.984 for
alpha 1 on the same data. Regularization is a dial, not a
virtue: too little leaves unstable coefficients, too much
erases the signal along with the noise. Day 1 in this repo
shows the unpenalized end of that same dial, where one outlier
bends the plain least squares line.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/ridge-regression/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
