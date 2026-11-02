"""Tests for PageRank: one that proves it works, one that breaks it."""

import numpy as np

from impl import PageRank, adjacency_from_edges


def test_works():
    """A 3-page cycle is perfectly symmetric, so every page must
    hold exactly one third of the rank, and ranks must sum to 1.
    On the demo site, Home (linked by every page) must rank first."""
    cycle = adjacency_from_edges(3, [(0, 1), (1, 2), (2, 0)])
    ranks = PageRank().fit(cycle).ranks_
    assert np.allclose(ranks, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)
    assert abs(ranks.sum() - 1.0) < 1e-9

    names_n = 6
    site = adjacency_from_edges(
        names_n,
        [(0, 1), (0, 2), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 4)],
    )
    ranks = PageRank().fit(site).ranks_
    assert abs(ranks.sum() - 1.0) < 1e-9
    assert np.all(ranks > 0)
    assert int(np.argmax(ranks)) == 0
    assert ranks[0] > ranks[1] and ranks[0] > ranks[2]


def test_dangling_page_keeps_rank():
    """A page with no outlinks must not drain rank or break the
    math: ranks still sum to 1 and every page keeps some rank."""
    adj = adjacency_from_edges(3, [(0, 1), (1, 2)])
    ranks = PageRank().fit(adj).ranks_
    assert abs(ranks.sum() - 1.0) < 1e-9
    assert np.all(ranks > 0)


def test_breaks():
    """With no teleportation (damping 1.0), rank gets trapped in a
    closed loop. Pages 0 and 1 link into a pair, pages 2 and 3,
    that only link to each other. After iteration, pages 0 and 1
    hold no rank at all and the loop holds all of it, even though
    pages 0 and 1 are real pages. The random jump is what keeps
    PageRank honest: it is the only way rank escapes a trap."""
    adj = adjacency_from_edges(4, [(0, 2), (1, 2), (2, 3), (3, 2)])
    ranks = PageRank(damping=1.0, max_iter=200).fit(adj).ranks_
    assert abs(ranks.sum() - 1.0) < 1e-9
    assert ranks[0] < 0.01 and ranks[1] < 0.01
    assert ranks[2] + ranks[3] > 0.99
