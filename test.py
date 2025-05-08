"""
test_max_mean_cycle
===================

Unit tests for `max_mean_cycle`.
Each test has a concise note explaining *what* and *why* it checks.
"""

import random
import math
import pytest

from max_mean_cycle import max_mean_cycle


# ---------------------------------------------------------------------------
# A tiny brute-force helper (exponential, only for n ≤ 8) – untouched
# ---------------------------------------------------------------------------
def brute_max_mean_cycle(n, edges):
    """Exhaustively enumerate simple cycles and return the maximum-mean one."""
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))

    best_mean = -math.inf
    best_cycle = None

    def dfs(start, v, visited, path, weight):
        nonlocal best_mean, best_cycle
        for w, wt in adj[v]:
            if w == start:
                mean = (weight + wt) / (len(path) + 1)
                if mean > best_mean:
                    best_mean = mean
                    best_cycle = path + [v]
            elif w not in visited and len(path) < 8:   # depth limit
                dfs(start, w, visited | {w}, path + [v], weight + wt)

    for v in range(n):
        dfs(v, v, {v}, [], 0.0)

    if best_cycle is None:
        raise ValueError("acyclic")
    return best_cycle, best_mean


# ---------------------------------------------------------------------------
# Hand-crafted deterministic cases
# ---------------------------------------------------------------------------

def test_simple_triangle():
    """The only cycle is 0→1→2→0, mean 5."""
    n = 3
    edges = [(0, 1, 5), (1, 2, 5), (2, 0, 5)]
    cycle, mean = max_mean_cycle(n, edges)
    assert pytest.approx(mean) == 5.0
    assert set(cycle) == {0, 1, 2}


def test_multiple_cycles():
    """
    Two disjoint cycles:
        * 0↔1   (mean 4)
        * 2→3→4→2 (mean 5, should be chosen)
    """
    edges = [
        (0, 1, 4), (1, 0, 4),
        (2, 3, 6), (3, 4, 4), (4, 2, 5)
    ]
    cycle, mean = max_mean_cycle(5, edges)
    assert pytest.approx(mean) == 5.0
    assert set(cycle) == {2, 3, 4}


def test_negative_weights():
    """
    Mix of positive and negative weights.
    Best-mean cycle is 2↔3 with mean 2 (beats 0↔1 whose mean is 1).
    """
    edges = [
        (0, 1, -1), (1, 0, 3),    # mean 1
        (2, 3, 2),  (3, 2, 2)     # mean 2
    ]
    cycle, mean = max_mean_cycle(4, edges)
    assert pytest.approx(mean) == 2.0
    assert set(cycle) == {2, 3}


# ---------------------------------------------------------------------------
# Randomised cross-check against brute force on tiny graphs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_against_bruteforce(seed):
    """
    For 20 random tiny graphs (n ≤ 7) compare our fast algorithm
    with an exponential brute-force enumerator: means must match.
    """
    random.seed(seed)
    n = random.randint(3, 7)
    edges = []
    for u in range(n):
        for v in range(n):
            if u != v and random.random() < 0.4:
                w = random.randint(-9, 9)
                edges.append((u, v, w))

    try:
        cyc_fast, mean_fast = max_mean_cycle(n, edges)
    except ValueError:
        # fast code says acyclic ⇒ brute must agree
        with pytest.raises(ValueError):
            brute_max_mean_cycle(n, edges)
    else:
        cyc_brute, mean_brute = brute_max_mean_cycle(n, edges)
        assert pytest.approx(mean_fast, abs=1e-6) == mean_brute
