"""Route reconstruction: per-observation arcs -> one drivable arc sequence.

The decoder returns ONE arc per GPS observation. Consecutive observations often sit on
arcs that are a few segments apart (at 1 Hz and 15 m/s a vehicle can cross several short
arcs between two fixes). Listing only the observed arcs therefore leaves holes in the
route: the connecting segments the vehicle must have driven are missing.

reconstruct_route() fixes that:
  1. drop -1 (observations without candidates) and collapse repeated arcs;
  2. for each consecutive pair (a, b) that is not directly connected
     (to_node(a) != from_node(b)), insert the shortest directed road path from the end
     of a to the start of b, found with Dijkstra and bounded by cutoff_m;
  3. if no path exists within cutoff_m (e.g. across a long GPS gap), the hole is left
     as it is - an honest break in the route, never an invented road.

The same function is applied to classifier-only and HMM output, so the two are compared
on equal terms. It works with RoadNetwork and with the light network objects used by the
HMM experiments (via the transitions helpers).
"""
from __future__ import annotations

import heapq
import math

from .transitions import _arc_endpoints, _arc_length, _out_arcs


def shortest_arc_path(network, start_node: int, end_node: int, cutoff_m: float) -> list[int] | None:
    """Arc ids of the shortest directed path start_node -> end_node, or None beyond cutoff_m."""
    if start_node == end_node:
        return []
    dist = {start_node: 0.0}
    via: dict[int, tuple[int, int]] = {}            # node -> (previous node, arc used)
    heap = [(0.0, start_node)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, math.inf):
            continue
        if node == end_node:
            path = []
            while node != start_node:
                node, arc = via[node]
                path.append(arc)
            return path[::-1]
        if d > cutoff_m:
            break
        for arc in _out_arcs(network, node):
            _, nxt = _arc_endpoints(network, arc)
            nd = d + _arc_length(network, arc)
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                via[nxt] = (node, int(arc))
                heapq.heappush(heap, (nd, nxt))
    return None


def reconstruct_route(arc_per_obs, network, cutoff_m: float = 2000.0) -> list[int]:
    """Ordered, de-duplicated arc sequence with connecting segments filled in."""
    arcs: list[int] = []
    for a in arc_per_obs:
        a = int(a)
        if a >= 0 and (not arcs or arcs[-1] != a):
            arcs.append(a)
    if len(arcs) < 2:
        return arcs
    route = [arcs[0]]
    for b in arcs[1:]:
        a = route[-1]
        _, end_a = _arc_endpoints(network, a)
        start_b, _ = _arc_endpoints(network, b)
        if end_a != start_b:
            path = shortest_arc_path(network, end_a, start_b, cutoff_m)
            if path:
                route.extend(path)
        if route[-1] != b:
            route.append(b)
    return route
