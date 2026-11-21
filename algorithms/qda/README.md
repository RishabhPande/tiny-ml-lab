# Day 31: Quadratic Discriminant Analysis (QDA)

## Idea

QDA fits one Gaussian per class and, unlike LDA, gives each class
its own covariance matrix. A point is scored under every class by
its Gaussian log posterior, which combines how far the point sits
from the class mean, measured against that class's own spread,
with a log determinant term for the size of that spread. Because
each class has its own quadratic term, the terms no longer cancel
between classes and the decision boundary can curve. That curve
lets QDA separate classes that share a mean and differ only in
spread, the exact case that leaves LDA near chance.

## Run

```bash
python algorithms/qda/impl.py
pytest algorithms/qda/test_qda.py
```

## Break it

`test_breaks` trains QDA on 10 points per class in 10 dimensions.
Each class covariance has 55 free numbers, so 10 points cannot pin
it down: the model scores perfectly on its training set and near
chance on a large test set from the same distribution. Trained on
400 points per class instead, the same setup generalizes far
better, which shows the failure is covariance estimation on tiny
samples, not the data. QDA buys its curved boundary with many more
parameters, and it needs the sample size to pay for them.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/qda/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
