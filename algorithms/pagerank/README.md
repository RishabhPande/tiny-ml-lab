# Day 12: PageRank

## Idea

PageRank ranks pages by link structure, not by content. Imagine a
surfer who follows a random outlink most of the time and jumps to a
random page the rest of the time. A page is important when important
pages link to it, so the ranking is the steady state of that walk:
start with equal rank everywhere, then repeat the walk until the
ranks stop moving. Dangling pages with no outlinks are treated as
linking everywhere, so rank never leaks out of the graph.

## Run

```bash
python algorithms/pagerank/impl.py
pytest algorithms/pagerank/test_pagerank.py
```

## Break it

`test_breaks` turns the random jump off (damping 1.0) on a graph
where two pages link only to each other. Those two pages swallow
all of the rank, and the pages that link into the loop end up with
nothing, even though they are real pages. The jump is not a detail:
without it, any closed loop is a trap. When a graph has link farms
or closed communities, raw link counting overrates them, and that
is exactly what the damping factor is there to stop.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/pagerank/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
