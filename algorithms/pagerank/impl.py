"""PageRank
Rank pages by the link structure of a small web graph: a page is
important when important pages link to it (numpy only).
Run: python algorithms/pagerank/impl.py
"""

import numpy as np


def adjacency_from_edges(n_pages, edges):
    """Build an adjacency matrix from (from_page, to_page) edge pairs.

    adj[i, j] is 1 when page i links to page j, else 0.
    """
    adj = np.zeros((n_pages, n_pages), dtype=float)
    for i, j in edges:
        adj[i, j] = 1.0
    return adj


class PageRank:
    """Random-surfer ranking by power iteration.

    The surfer follows a random outlink with probability `damping`
    and jumps to a uniformly random page otherwise. Dangling pages
    (no outlinks) are treated as linking to every page, so rank
    never leaks out of the graph. With damping 1.0 there is no
    random jump, and rank can get trapped in a closed loop of
    pages that never link out: see test_breaks.
    """

    def __init__(self, damping=0.85, max_iter=100, tol=1e-6):
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def _transition(self, adj):
        """Column-stochastic matrix: column i spreads page i's rank
        over its outlinks, or uniformly when page i is dangling."""
        n = adj.shape[0]
        out_degree = adj.sum(axis=1)
        trans = np.zeros((n, n), dtype=float)
        for i in range(n):
            if out_degree[i] > 0:
                trans[:, i] = adj[i] / out_degree[i]
            else:
                trans[:, i] = 1.0 / n
        return trans

    def fit(self, adj):
        """Rank the pages in adjacency matrix adj.

        Sets ranks_ (sums to 1) and n_iter_ (iterations run).
        """
        adj = np.asarray(adj, dtype=float)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError("adj must be a square matrix")
        if np.any(adj < 0):
            raise ValueError("adj must not contain negative entries")
        n = adj.shape[0]
        trans = self._transition(adj)
        ranks = np.full(n, 1.0 / n)
        teleport = (1.0 - self.damping) / n
        self.n_iter_ = 0
        for it in range(1, self.max_iter + 1):
            new_ranks = self.damping * (trans @ ranks) + teleport
            self.n_iter_ = it
            if np.sum(np.abs(new_ranks - ranks)) < self.tol:
                ranks = new_ranks
                break
            ranks = new_ranks
        self.ranks_ = ranks / ranks.sum()
        return self


def _demo_graph():
    """A six-page site where every page links to Home, and Home
    links out to Blog and Docs. Names are in page-index order."""
    names = ["Home", "Blog", "Docs", "About", "Contact", "FAQ"]
    edges = [
        (0, 1),
        (0, 2),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (5, 4),
    ]
    return names, adjacency_from_edges(len(names), edges)


if __name__ == "__main__":
    names, adj = _demo_graph()
    model = PageRank(damping=0.85).fit(adj)
    print(f"PageRank converged in {model.n_iter_} iterations.")
    print(f"Ranks sum to {model.ranks_.sum():.6f}.")
    order = np.argsort(-model.ranks_)
    for rank, i in enumerate(order, start=1):
        print(f"{rank}. {names[i]}: {model.ranks_[i]:.4f}")
