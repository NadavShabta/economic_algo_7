"""
max_mean_cycle
==============

Find the directed cycle with **maximum average weight** in a weighted graph.

Implementation details
----------------------
* Kosaraju → strongly-connected components
* Karp’s dynamic-programming algorithm → maximum-mean cycle per SCC
* Overall runtime:  O(|V| · |E|)
* Overall memory:   O(|V|)  (rows are rolled)

The public entry-point is :func:`max_mean_cycle`.
"""

from collections import defaultdict
from math import inf
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Type aliases (for readability only)
# ---------------------------------------------------------------------------

Vertex = int
Edge   = Tuple[int, int, float]     # (u, v, weight)


# ---------------------------------------------------------------------------
# Strongly-connected components  (Kosaraju, two DFS passes)
# ---------------------------------------------------------------------------

def _scc(n: int,
         g_out: List[List[int]],
         g_in:  List[List[int]]) -> List[List[int]]:
    """Return a list of SCCs, each given as *a list of its vertices*."""
    seen, order = [False] * n, []

    def dfs1(v: int) -> None:
        seen[v] = True
        for w in g_out[v]:
            if not seen[w]:
                dfs1(w)
        order.append(v)

    for v in range(n):
        if not seen[v]:
            dfs1(v)

    comp  = [-1] * n
    comps: List[List[int]] = []

    def dfs2(v: int, cidx: int) -> None:
        comp[v] = cidx
        comps[cidx].append(v)
        for w in g_in[v]:
            if comp[w] == -1:
                dfs2(w, cidx)

    for v in reversed(order):
        if comp[v] == -1:
            comps.append([])
            dfs2(v, len(comps) - 1)

    return comps


# ---------------------------------------------------------------------------
# Karp’s algorithm on **one** SCC
# ---------------------------------------------------------------------------

def _karp_on_vertices(n: int,
                      edges: List[Edge],
                      vertices: List[int]) -> Tuple[List[int], float]:
    """
    Run Karp on the induced sub-graph `vertices`.

    Returns
    -------
    cycle_vertices : List[int]
        The cycle achieving the maximum mean (in forward order).
    mean_weight : float
        The value of that maximum mean.
    """
    # map vertex → 0…m-1 for compact tables
    idx = {v: i for i, v in enumerate(vertices)}
    m   = len(vertices)

    re_edges = [(idx[u], idx[v], w)
                for (u, v, w) in edges
                if u in idx and v in idx]
    if not re_edges:
        raise ValueError("component without edges")

    # DP[k][v] – best weight of a length-k path ending at v
    dp   = [[-inf] * m for _ in range(m + 1)]
    pred = [[-1   ] * m for _ in range(m + 1)]
    for v in range(m):
        dp[0][v] = 0.0

    for k in range(1, m + 1):
        for u, v, w in re_edges:
            if dp[k - 1][u] == -inf:
                continue
            cand = dp[k - 1][u] + w
            if cand > dp[k][v]:
                dp[k][v]   = cand
                pred[k][v] = u

    # pick the vertex with the best (maximum) mean
    best_mean, best_v, best_k = -inf, None, None
    for v in range(m):
        if dp[m][v] == -inf:
            continue
        worst, arg_k = inf, -1
        for k in range(m):
            if dp[k][v] == -inf:
                continue
            mean = (dp[m][v] - dp[k][v]) / (m - k)
            if mean < worst:
                worst, arg_k = mean, k
        if worst > best_mean:
            best_mean, best_v, best_k = worst, v, arg_k

    if best_v is None:
        raise ValueError("component is acyclic")

    # ------------------------------------------------------------------
    # Extract *exactly* the cycle — classic “first-repeat” trick
    # ------------------------------------------------------------------
    order, first_seen = [], {}
    row, v = m, best_v

    for _ in range(m + 1):            # ≤ m predecessors exist
        if v in first_seen:           # found the repeat – slice is the cycle
            cycle_idx = order[first_seen[v]:]
            break
        first_seen[v] = len(order)
        order.append(v)
        v = pred[row][v]
        row -= 1                      # move up one DP row each step

    cycle_idx.reverse()               # we built it backwards
    cycle = [vertices[x] for x in cycle_idx]
    return cycle, best_mean


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def max_mean_cycle(n: int, edges: List[Edge]) -> Tuple[List[Vertex], float]:
    """
    Find the directed cycle with the **maximum average edge weight**.

    Parameters
    ----------
    n : int
        Number of vertices (assumed labelled 0…n-1).
    edges : List[Edge]
        Each edge is a tuple ``(u, v, weight)``.

    Returns
    -------
    (cycle, mean) : Tuple[List[int], float]
        *cycle* – vertices in forward order (one full lap, no repeat)\
        *mean*   – the cycle’s average weight.

    Raises
    ------
    ValueError
        If the graph is acyclic.
    """
    # build adjacency lists for Kosaraju
    g_out, g_in = [[] for _ in range(n)], [[] for _ in range(n)]
    for u, v, _ in edges:
        g_out[u].append(v)
        g_in[v].append(u)

    best_cycle, best_mean = None, -inf
    for comp in _scc(n, g_out, g_in):
        if len(comp) == 1 and not g_out[comp[0]]:   # isolated vertex
            continue
        try:
            cycle, mean = _karp_on_vertices(n, edges, comp)
            if mean > best_mean:
                best_cycle, best_mean = cycle, mean
        except ValueError:
            # component was acyclic – just skip it
            pass

    if best_cycle is None:
        raise ValueError("The input graph has no directed cycle.")
    return best_cycle, best_mean

