# Day 9: Random Forest

## Idea

Grow many decision trees and let them vote. Each tree trains on a bootstrap
resample of the rows and, at each split, sees only a random subset of the
features. Bagging plus random features decorrelates the trees, so their
individual quirks average out and the vote is steadier than any one tree.
Feature importances come from adding up each split's impurity drop.

## Run

```bash
python algorithms/random-forest/impl.py
pytest algorithms/random-forest/test_random-forest.py
```

## Break it

`test_breaks` builds a rare, overlapping class: 5 percent of the data, sitting
close to the majority blob. Its leaves are rarely pure, so the majority vote
almost never picks it, and held-out recall for that class is 0. Overall
accuracy still looks fine, which is the trap: a single headline number hides
the class the model never predicts. When a rare class matters, check recall
per class, and reach for class weights, resampling, or a threshold tuned to
that class instead of trusting accuracy.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/random-forest/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
