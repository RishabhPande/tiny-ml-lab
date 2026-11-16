# Day 26: Elastic Net

## Idea

The elastic net (Zou and Hastie, 2005) is least squares with
both of the penalties from the last two days at once: the L1
penalty of the lasso (Day 25), which can hold a coefficient at
exactly zero, and the L2 penalty of ridge (Day 24), which
shrinks smoothly and shares an effect across correlated
features. One knob, l1_ratio, sets the mix: 1.0 is the lasso,
0.0 is ridge, and 0.5 sits in between. It is fitted here by
coordinate descent, where each update is a soft-threshold
divided by the squared column size plus the L2 strength. On
the grouped demo here, three features are near copies of one
factor and the truth splits the effect evenly, (1.5, 1.5, 1.5).
The elastic net recovers (1.474, 1.444, 1.470), sets all four
noise features to exactly 0, and scores held-out R2 of 0.988.
The lasso on the same data splits the group (1.684, 1.148,
1.576), a spread of 0.537 against the elastic net's 0.030.

## Run

```bash
python algorithms/elastic-net/impl.py
pytest algorithms/elastic-net/test_elastic-net.py
```

## Break it

`test_breaks` keeps the data and changes only the units. The
one real signal is stored divided by 100, as happens when a
column is recorded in centimeters instead of meters, so its
true coefficient is 300 instead of 3. The penalties price the
coefficient, not the effect it has, so a coefficient that
large is unaffordable: the fit sets the only real feature to
exactly 0 and held-out R2 lands at 0.000, as if there were no
signal at all. Multiply the same column back up by 100, which
changes nothing about the information it carries, and the
same settings recover a coefficient of 2.846 with held-out R2
of 0.972. Penalized fits are only meaningful on features that
have been put on comparable scales first, and this is the
test that shows why: skip the scaling step and the method
does not degrade, it deletes the signal and reports nothing
wrong.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/elastic-net/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
