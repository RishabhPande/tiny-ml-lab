# Day 38: Learning Vector Quantization

## Idea

Learning vector quantization keeps a few labelled prototypes
per class instead of the whole training set. Training draws
one point at a time, finds its closest prototype, and moves
that prototype toward the point when the labels match or away
from it when they do not, with the step size shrinking over
time. A new point is classified by its closest prototype, so
the prototypes carve the space into labelled regions, a bit
like K-nearest neighbors with a tiny learned summary in place
of the full data.

## Run

```bash
python algorithms/learning-vector-quantization/impl.py
pytest algorithms/learning-vector-quantization/test_learning_vector_quantization.py
```

## Break it

`test_breaks` puts class 0 in two clusters at (-4, 0) and
(4, 0) with class 1 in the middle at (0, 0). With only one
prototype per class, the class 0 prototype can guard just one
of its two clusters, the other cluster falls to the class 1
prototype, and accuracy stays below 0.80. Two prototypes per
class give each cluster its own guard and accuracy rises
above 0.95. The missing piece is a prototype, not more
training.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/learning-vector-quantization/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
