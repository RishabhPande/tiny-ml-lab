# Day 21: Isolation Forest

## Idea

Normal points live in dense regions, so a random split rarely
cuts one of them off from the crowd. A lone odd point gets
isolated in just a few splits. Each tree in the forest picks a
random feature and a random split between its min and max, over
and over, until every point in its subsample sits alone or a
depth limit is hit. Points with a short average path length
across the forest are the anomalies, rescaled to a score near
1; ordinary points score near 0.5.

## Run

```bash
python algorithms/isolation-forest/impl.py
pytest algorithms/isolation-forest/test_isolation-forest.py
```

## Break it

`test_breaks` shows masking. Put 100 anomalies in one tight
cluster at (5, 5) and their mean score drops to about 0.49, no
higher than the normal points, and fewer than half are flagged
even when 100 points are flagged. Scatter the same 100
anomalies alone at the same distance from the blob and recall
is perfect. The first splits separate the whole cluster from
the normal data in one cut, but each member still needs
several more splits to be isolated from its neighbours, and
those extra splits erase the short path that would have given
it away. A fraud ring that acts together can hide the same
way: lone outliers are easy, coordinated groups are not.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/isolation-forest/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
