# Day 13: Support Vector Machine (SVM)

## Idea

An SVM draws the widest street it can between two classes. Each
training point votes for its side, and the boundary sits halfway
between the closest points on either side, the support vectors.
Training minimizes hinge loss (zero for points safely on the right
side of the street, growing for points inside it or across it) plus
a penalty on the size of the weights, which is what keeps the street
wide. This implementation uses full-batch subgradient descent: each
step shrinks the weights slightly toward zero, then nudges them
toward the points that are inside the margin or wrong.

## Run

```bash
python algorithms/svm/impl.py
pytest algorithms/svm/test_svm.py
```

## Break it

`test_breaks` trains the SVM on XOR: two diagonal corners labeled
+1, the other two labeled -1. No straight line can separate that
pattern, so the SVM stalls near 75 percent accuracy even though the
same code scores near 100 percent on separated blobs. The optimizer
is fine, the model class is the limit. That is the cue to reach for
a kernel, which bends the boundary by mapping the data into a space
where a straight line works again.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/svm/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
