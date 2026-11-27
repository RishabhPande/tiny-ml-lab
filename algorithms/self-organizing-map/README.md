# Day 37: Self-Organizing Map

## Idea

A self-organizing map is a grid of prototypes that learns to
spread over the data. Training finds each point's closest
prototype, the best matching unit, and pulls that unit and
its grid neighbours toward the point, with the pull shrinking
over time and with grid distance. Because neighbours move
together, nearby points in the data land on the same unit or
on neighbouring units, so the flat grid ends up as a readable
2D map of data that may have many more dimensions.

## Run

```bash
python algorithms/self-organizing-map/impl.py
pytest algorithms/self-organizing-map/test_self_organizing_map.py
```

## Break it

`test_breaks` trains a 1 by 1 grid on three far apart blobs.
One prototype has to stand in for all three clusters at once,
so the quantization error stays above 1.5 where an 8 by 8 map
scores below 0.35, and every point is assigned to the same
single unit. No training schedule fixes a grid that is too
small; the capacity is missing, not the optimisation.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/self-organizing-map/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
