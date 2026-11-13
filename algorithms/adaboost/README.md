# Day 23: AdaBoost

## Idea

AdaBoost (Freund and Schapire, 1997) builds a strong classifier
out of weak ones, here decision stumps that split on one feature
at one threshold. Each round fits the stump that does best on the
weighted data, gives it a vote based on its accuracy, and raises
the weights of the points the ensemble still gets wrong. The
final call is the sign of the weighted vote. On clean data the
weak learners correct each other and the ensemble is far better
than any single stump.

## Run

```bash
python algorithms/adaboost/impl.py
pytest algorithms/adaboost/test_adaboost.py
```

## Break it

`test_breaks` flips 30 percent of the training labels at random
and trains on the noisy set. The flipped points sit deep on the
wrong side, so they are the hardest points in every round, their
weights keep growing, and later stumps bend toward the noise
instead of the boundary. On clean held-out data the boosted
model scores 0.830, worse than a single stump trained on the
same noisy data at 0.930, and far worse than boosting trained
on clean labels at 0.965. Gradient boosting in this repo has
the same family weakness in a different shape: it fits whatever
the labels say, so bad labels become the target.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/adaboost/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
