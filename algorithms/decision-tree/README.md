# Day 5: Decision Tree

Split the data on the feature and threshold that most reduces Gini
impurity, recurse on each side, and stop at a depth limit or when a node
is pure. The whole model is a set of axis-aligned if-else rules you can
read out loud.

## Run

```bash
python algorithms/decision-tree/impl.py
pytest algorithms/decision-tree/test_decision-tree.py
```

## Do it yourself

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/decision-tree/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.

## What the break-it test shows

`test_breaks` flips 20 percent of the training labels and removes the depth
limit. The tree keeps splitting until every leaf is pure, which means it
memorizes the noise: near-perfect train accuracy and a big gap to test
accuracy. Depth limits, pruning, and ensembles exist because an unlimited
tree would rather memorize than generalize.
