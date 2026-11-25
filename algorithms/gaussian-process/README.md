# Day 35: Gaussian Process Regression

## Idea

A Gaussian process regressor fits a distribution over curves
instead of one curve. A kernel says how strongly two points
should move together: close points, similar values. Solving
one linear system built from the kernel blends the training
values into a prediction at any new point, and the same math
gives a standard deviation that stays small near the data and
grows away from it, so the model knows where it is guessing.

## Run

```bash
python algorithms/gaussian-process/impl.py
pytest algorithms/gaussian-process/test_gaussian_process.py
```

## Break it

`test_breaks` runs length scale 10.0 on a sine wave spanning
only [0, 6]. Every distance is tiny next to the length scale,
so the kernel matrix goes nearly constant, the weights stop
depending on position, and the prediction flattens toward the
training mean: R2 drops below 0.3 where length scale 1.0
scores above 0.95. The length scale is the model's idea of
how fast the curve may bend, and no amount of data fixes a
kernel that cannot tell near from far.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/gaussian-process/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
