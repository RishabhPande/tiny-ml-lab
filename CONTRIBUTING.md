# Contributing to Tiny ML Lab

Everyone is welcome. The contribution bar is deliberately low: one algorithm
is one folder, and one folder is four files.

## Adding a new algorithm

Pick an algorithm from [REQUESTS.md](REQUESTS.md) (or propose a new one in
an issue first), then add a folder `algorithms/<slug>/` with exactly these
files:

1. `impl.py`: the full implementation, numpy only, with a standalone demo
   under `__main__` that runs in under 60 seconds on synthetic data with a
   fixed seed and no downloads.
2. `workbook.py`: the same public API as `impl.py`, but with TODO bodies
   that raise `NotImplementedError`, plus a standalone self-check under
   `__main__` that prints which parts are still TODO. No pytest dependency.
   Name it `workbook.py` so pytest never collects it.
3. `test_<slug>.py`: pytest tests with at least `test_works` (a real
   numeric assertion that the method works) and `test_breaks` (a real
   failure mode, not a toy error).
4. `README.md`: the idea in a few lines, the run commands, and what the
   break-it test shows. Also explain the fork mechanic: fork the repo, fill
   in the workbook, run it to check yourself, then peek at `impl.py`.

Then update two root files:

5. Add one row to the days table in `README.md`.
6. Add one checkbox row to [PROGRESS.md](PROGRESS.md).

## Rules

- Numpy only. No scikit-learn, no torch, no downloads, no datasets.
- Every demo runs in under 60 seconds.
- Every test must be a real numeric assertion, not `assert True`.
- No em-dashes in code, comments, or docs. Use plain hyphens.
- Run `pytest` from the repo root before opening a PR. All tests must pass.

## Pull requests

Fork the repo, add your folder, run the full suite, and open a PR with a
one-line description of the algorithm and its break-it test. Small, focused
PRs get merged fastest. If you completed days in your fork, add yourself to
[SHOWCASE.md](SHOWCASE.md) in the same PR or a separate one.
