"""Label-method comparison: monotonic DP alignment (labels.align_route) vs nearest-route-arc
(labels.nearest_route_arc). Evidence for the report/viva on why DP labelling is used.
Run from repo root:
    set PYTHONPATH=src
    python scripts/compare_label_methods.py [split/trip ...]
Default trips: train/00000054, train/00000033 (offset trips), train/00000001 (clean), student_test/00000052 (has a loop).
Writes experiments/results/eda_label_method_comparison.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from pathsnap import io
from pathsnap.geo import LocalProjection, point_to_segment
from pathsnap.network import RoadNetwork
from pathsnap.labels import align_route, nearest_route_arc

args = sys.argv[1:] or ["train/00000054", "train/00000033", "train/00000001", "student_test/00000052"]
rows = []
for a in args:
    split, trip = a.split("/")
    d = ROOT / "data" / "raw" / split / trip
    trk = io.load_track(d)
    nodes, arcs = io.load_network(d)
    net = RoadNetwork(nodes, arcs, LocalProjection.from_track(trk))
    route = io.load_route(d)
    px, py = net.proj.to_xy(trk["lon"].to_numpy(), trk["lat"].to_numpy())

    j_dp, d_dp = align_route(px, py, route, net)        # route position (non-decreasing), distance
    j_nn = nearest_route_arc(px, py, route, net)         # route position of nearest arc, no time order
    d_nn, _ = point_to_segment(px, py, net.x1[route[j_nn]], net.y1[route[j_nn]],
                               net.x2[route[j_nn]], net.y2[route[j_nn]])
    diff = route[j_dp] != route[j_nn]
    back_nn = int((np.diff(j_nn) < 0).sum())             # times nearest-arc labelling jumps backwards along the route
    back_dp = int((np.diff(j_dp) < 0).sum())             # must be 0 by construction
    rows.append(dict(trip=trip, partition=split, n_obs=len(px), route_len=len(route),
                     pct_labels_differ=round(100 * diff.mean(), 2),
                     mean_dist_dp_m=round(float(np.mean(d_dp)), 2), mean_dist_nearest_m=round(float(np.mean(d_nn)), 2),
                     p95_dist_dp_m=round(float(np.percentile(d_dp, 95)), 1), p95_dist_nearest_m=round(float(np.percentile(d_nn, 95)), 1),
                     backward_jumps_dp=back_dp, backward_jumps_nearest=back_nn))
    print(rows[-1])

out = pd.DataFrame(rows)
p = ROOT / "experiments" / "results" / "eda_label_method_comparison.csv"
p.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(p, index=False)
print("\n" + out.to_string(index=False))
print("saved", p)
