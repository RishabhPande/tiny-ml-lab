# Day 8: Logistic Regression

## Idea

Fit a straight-line boundary by gradient descent on the log loss: turn a
linear score into a probability with the sigmoid, then nudge the weights
to make the true labels more probable. The learned weights are readable:
each one says how much its feature pushes the odds toward class 1.

## Run

```bash
python algorithms/logistic-regression/impl.py
pytest algorithms/logistic-regression/test_logistic-regression.py
```

## Break it

`test_breaks` builds XOR: class 1 sits on two opposite corners, class 0 on
the other two. No single straight line can separate that pattern, so
accuracy stalls near chance no matter how long you train. When your
boundary needs a bend, a linear model is the wrong tool: add features,
or switch to a model that can curve.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/logistic-regression/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
