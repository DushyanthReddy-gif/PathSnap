"""
Transition log-probabilities for the HMM decoder.

Design (Newson & Krumm 2009 §3.2)
-----------------------------------
Given consecutive candidate arcs  a  (at time t-1) and  b  (at time t):

    log P(a → b) ∝ -|route_distance(a, b) - gps_distance| / beta

where
  gps_distance  = Euclidean distance between successive GPS points (metres),
  route_distance = shortest-path distance on the road network (metres),
  beta           = tuning parameter (metres).

Special cases
~~~~~~~~~~~~~
* Same arc (a == b): allowed; log-penalty is small positive (near 0).
* Directly connected (to_node(a) == from_node(b)): route_dist = arc b length.
* Disconnected within cutoff_m: bounded Dijkstra gives exact route_dist.
* Unreachable (Dijkstra exhausted, or cost > cutoff_m): return LARGE_PENALTY
  (a large finite number, NOT -inf, so Viterbi can still continue).

Caching
~~~~~~~
Shortest-path results are cached per (from_arc, to_arc) pair in a module-level
dict keyed by the network identity (id(network)).  On a single trip the cache
avoids recomputing the same pair when multiple consecutive observations hit the
same candidate arcs.
"""

from __future__ import annotations

import heapq
import math
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Finite penalty for unreachable transitions (large but not -inf).
LARGE_PENALTY: float = -1_000.0

# Cache: id(network) → {(from_arc, to_arc): route_dist_m}
_sp_cache: dict[int, dict[tuple[int, int], float]] = {}


def _get_cache(network) -> dict[tuple[int, int], float]:
    nid = id(network)
    if nid not in _sp_cache:
        _sp_cache[nid] = {}
    return _sp_cache[nid]


def clear_cache(network=None) -> None:
    """Clear the shortest-path cache (call when switching trips)."""
    global _sp_cache
    if network is None:
        _sp_cache.clear()
    else:
        _sp_cache.pop(id(network), None)


# ---------------------------------------------------------------------------
# Network helpers (work with either a RoadNetwork object or plain dicts)
# ---------------------------------------------------------------------------

def _arc_endpoints(network, arc_id: int) -> tuple[int, int]:
    """Return (from_node, to_node) for arc_id."""
    # RoadNetwork stores arcs_df; fall back to plain attribute access.
    try:
        row = network.arcs_df.iloc[arc_id]
        return int(row["from_node"]), int(row["to_node"])
    except AttributeError:
        return network.from_node[arc_id], network.to_node[arc_id]


def _arc_length(network, arc_id: int) -> float:
    """Return precomputed length_m of arc_id."""
    try:
        return float(network.arcs_df.iloc[arc_id]["length_m"])
    except AttributeError:
        return float(network.length_m[arc_id])


def _out_arcs(network, node_id: int) -> list[int]:
    """Return list of arc IDs leaving node_id."""
    try:
        return network.out_arcs[node_id]
    except (AttributeError, KeyError):
        return []


# ---------------------------------------------------------------------------
# Bounded Dijkstra
# ---------------------------------------------------------------------------

def _dijkstra(network, start_node: int, end_node: int, cutoff_m: float) -> float:
    """
    Shortest-path distance (metres) on the directed graph from start_node
    to end_node.  Stops once the settled distance > cutoff_m.

    Returns
    -------
    float: route distance in metres, or math.inf if unreachable within cutoff.
    """
    if start_node == end_node:
        return 0.0

    dist: dict[int, float] = {start_node: 0.0}
    heap: list[tuple[float, int]] = [(0.0, start_node)]

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, math.inf):
            continue
        if node == end_node:
            return d
        if d > cutoff_m:
            break
        for arc_id in _out_arcs(network, node):
            _, to_node = _arc_endpoints(network, arc_id)
            new_d = d + _arc_length(network, arc_id)
            if new_d < dist.get(to_node, math.inf):
                dist[to_node] = new_d
                heapq.heappush(heap, (new_d, to_node))

    return math.inf


# ---------------------------------------------------------------------------
# Route distance between two candidate arcs
# ---------------------------------------------------------------------------

def route_distance(network, arc_a: int, arc_b: int, cutoff_m: float) -> float:
    """
    Shortest road-network distance (metres) from arc_a to arc_b.

    If arc_a == arc_b the distance is 0.
    Otherwise Dijkstra is run from to_node(arc_a) to from_node(arc_b),
    then arc_b length is added.
    """
    if arc_a == arc_b:
        return 0.0

    cache = _get_cache(network)
    key = (arc_a, arc_b)
    if key in cache:
        return cache[key]

    _, start = _arc_endpoints(network, arc_a)   # to_node of arc_a
    end, _   = _arc_endpoints(network, arc_b)   # from_node of arc_b

    d_sp = _dijkstra(network, start, end, cutoff_m)
    if d_sp < math.inf:
        d_sp += _arc_length(network, arc_b)
    # Cap cached value so it doesn't blow memory on huge networks
    cache[key] = d_sp
    return d_sp


# ---------------------------------------------------------------------------
# Transition log-probability
# ---------------------------------------------------------------------------

def transition_log_prob(
    network,
    arc_a: int,
    arc_b: int,
    gps_dist_m: float,
    beta: float,
    cutoff_m: float,
) -> float:
    """
    log P(a → b) for one HMM transition step.

    Parameters
    ----------
    network   : RoadNetwork (or compatible object).
    arc_a     : candidate arc at time t-1.
    arc_b     : candidate arc at time t.
    gps_dist_m: Euclidean distance between consecutive GPS points.
    beta      : smoothness parameter (metres).  Smaller = prefers matching dist.
    cutoff_m  : Dijkstra cutoff (metres).  Pairs beyond this get LARGE_PENALTY.

    Returns
    -------
    float: log-transition probability (always ≤ 0, or LARGE_PENALTY).
    """
    d_route = route_distance(network, arc_a, arc_b, cutoff_m)
    if d_route == math.inf:
        return LARGE_PENALTY
    diff = abs(d_route - gps_dist_m)
    return -diff / beta


# ---------------------------------------------------------------------------
# Build full transition matrix for a step
# ---------------------------------------------------------------------------

def build_transition_matrix(
    network,
    arcs_prev: list[int],
    arcs_curr: list[int],
    gps_dist_m: float,
    beta: float,
    cutoff_m: float,
) -> np.ndarray:
    """
    Return a (len(arcs_prev), len(arcs_curr)) matrix of log transition probs.
    Rows = previous states, columns = current states.
    """
    n, m = len(arcs_prev), len(arcs_curr)
    mat = np.full((n, m), LARGE_PENALTY, dtype=np.float64)
    for i, a in enumerate(arcs_prev):
        for j, b in enumerate(arcs_curr):
            mat[i, j] = transition_log_prob(network, a, b, gps_dist_m, beta, cutoff_m)
    return mat
