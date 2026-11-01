# Day 11: Gradient Boosting

## Idea

Gradient boosting starts at the mean of y and then corrects itself,
one small step at a time. Each round fits a decision stump, a tree
with a single split, to the current residuals, the errors the model
still makes. The stump's predictions are shrunk by the learning rate
and added to the model, so the next round chases a smaller leftover
error. One stump is a crude step function, but a hundred shrunken
stumps stacked together trace out a smooth curve. It is the opposite
of a random forest: instead of many independent trees voting, each
tree here is built to fix the mistakes of the ones before it.

## Run

```bash
python algorithms/gradient-boosting/impl.py
pytest algorithms/gradient-boosting/test_gradient_boosting.py
```

## Break it

`test_breaks` trains on a straight line, y = x, with x between 0 and
1, then asks for x = 5. The true answer is 5, but the model predicts
something near 1. Every stump is a step: past its threshold it returns
one fixed value, no matter how far past it you go. So a boosted tree
model is flat outside the range it trained on, and it can never
continue a trend it has not seen. When the test data may run past the
training range, reach for a model that extrapolates, like linear
regression, or make sure the training data already covers that range.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/gradient-boosting/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
