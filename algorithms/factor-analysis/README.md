# Day 36: Factor Analysis

## Idea

Factor analysis says each measured variable is a shared part
plus a private part. A few hidden factors, mixed by the
loadings, make the shared part, and each variable also gets
its own noise variance, so a mostly-noise variable does not
drag a factor toward itself the way it can in PCA. Fitting
runs by expectation-maximisation: guess the factors for each
point, refit the loadings and noise variances from the
guesses, and repeat until the loadings settle.

## Run

```bash
python algorithms/factor-analysis/impl.py
pytest algorithms/factor-analysis/test_factor_analysis.py
```

## Break it

`test_breaks` asks for one factor on data built from two
independent, equally strong factors. The single factor can
follow only one direction at a time, so the other factor's
share of the signal has nowhere to go: reconstruction R2
stays below 0.65 where the two-factor fit scores above 0.85.
More data does not fix a factor count that is too small.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/factor-analysis/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
