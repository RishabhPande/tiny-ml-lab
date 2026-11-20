# Day 30: Linear Discriminant Analysis (LDA)

## Idea

LDA looks for projection directions where the class means sit far
apart and each class stays tight. It builds a within-class scatter
matrix from the spread around each class mean and a between-class
scatter matrix from how the means spread around the overall mean,
then solves for the directions that maximize the ratio of the two.
Points are classified by the nearest class centroid in the
projected space. Because only means and a shared spread enter the
math, LDA is fast, stable, and blind to any difference that leaves
the means in the same place.

## Run

```bash
python algorithms/lda/impl.py
pytest algorithms/lda/test_lda.py
```

## Break it

`test_breaks` builds two classes with the exact same mean: a tight
ball at the center and a wide ball around it. The between-class
scatter is near zero, the top eigenvalue collapses, and accuracy
stays close to chance. A simple rule based on distance from the
center separates the same data well, which shows the failure is
LDA's mean-only view, not the data. Classes that differ in spread,
shape, or ring structure need a method that can see more than
centroids.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/lda/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
