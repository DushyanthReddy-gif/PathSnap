"""Turn the reference route into ONE true arc per GPS observation.

PROBLEM: `.route` is an ordered list of arcs (no timestamps), but the models need
a "correct arc" for every GPS point.

IDEA (monotonic alignment): GPS points are in time order and the route is in
travel order, so as time goes forward we may stay on the same route arc or move
FORWARD along the route, but never go backwards.  Among all such assignments we
pick the one with the smallest total distance between each GPS point and its
assigned route arc.  This is a small dynamic-programming problem:

    best[i][j] = cost(i, j) + min over j' <= j of best[i-1][j']

    cost(i, j) = distance (metres) from GPS point i to route arc j, capped at CAP_M
                 so that one wild GPS point cannot dominate the total.

Why not simply "nearest route arc"?  On loops, U-turns and parallel/return streets
the nearest route arc can be one the car passes at a totally different time; the
monotonic constraint prevents that.

LEAKAGE RULE: this module reads the reference route.  Only training / evaluation
code may import it.  The prediction path (candidates, features, model, hmm) must not.
"""
import numpy as np

from . import io
from .geo import point_to_segment

CAP_M = 200.0


def _route_geometry(route, network):
    return (network.x1[route], network.y1[route], network.x2[route], network.y2[route])


def align_route(px, py, route, network, cap_m=CAP_M):
    """Monotonic alignment. Returns (route_idx, dist_m): for each point i, the position
    in `route` it is assigned to (non-decreasing), and its distance to that arc."""
    n, m = len(px), len(route)
    rx1, ry1, rx2, ry2 = _route_geometry(route, network)
    col = np.arange(m)

    back = np.empty((n, m), dtype=np.int32 if m > 32000 else np.int16)  # argmin pointers
    prev = None
    for i in range(n):
        d, _ = point_to_segment(px[i], py[i], rx1, ry1, rx2, ry2)
        cost = np.minimum(d, cap_m)
        if prev is None:
            cur = cost
        else:
            pm = np.minimum.accumulate(prev)                      # best cost up to column j
            back[i] = np.maximum.accumulate(np.where(prev == pm, col, 0))  # which column gave it
            cur = cost + pm
        prev = cur

    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        j[i - 1] = back[i, j[i]]

    d_assigned, _ = point_to_segment(px, py, rx1[j], ry1[j], rx2[j], ry2[j])
    return j, d_assigned


def nearest_route_arc(px, py, route, network, chunk=500):
    """Simple baseline for comparison: each point takes its nearest route arc, ignoring time order."""
    rx1, ry1, rx2, ry2 = _route_geometry(route, network)
    out = np.empty(len(px), dtype=np.int64)
    for s in range(0, len(px), chunk):
        d, _ = point_to_segment(px[s:s + chunk, None], py[s:s + chunk, None],
                                rx1[None], ry1[None], rx2[None], ry2[None])
        out[s:s + chunk] = d.argmin(axis=1)
    return out


def label_trip(trip_dir, track, network):
    """Per-observation labels for one trip.

    Returns a DataFrame indexed by obs_idx with columns:
        true_arc_id   the arc_id (row in .arcs) assigned by the alignment
        route_idx     position of that arc inside the route
        label_dist_m  distance from the GPS point to its assigned arc
    """
    import pandas as pd
    route = io.load_route(trip_dir)
    px, py = network.proj.to_xy(track["lon"].to_numpy(), track["lat"].to_numpy())
    j, d = align_route(px, py, route, network)
    return pd.DataFrame({"true_arc_id": route[j].astype("int32"),
                         "route_idx": j.astype("int32"),
                         "label_dist_m": d.astype("float32")}, index=track.index)
