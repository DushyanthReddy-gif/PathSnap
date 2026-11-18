"""Check labels.py on a few trips. Usage (repo root): python scripts/test_labels.py data/raw"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from pathsnap import io
from pathsnap.geo import LocalProjection
from pathsnap.network import RoadNetwork
from pathsnap.labels import label_trip, align_route, nearest_route_arc

root = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "data" / "raw")  # default: repo data/raw
train = f"{root}/train"
trips = io.list_trips(train)
npts = {t: sum(1 for _ in open(f"{train}/{t}/{t}.track")) for t in trips}
pick = [min(npts, key=npts.get), trips[10], trips[40], max(npts, key=npts.get)]
print("trip, n_obs, route_len, seconds, mean_dist, p95_dist, monotonic, agree_with_nearest")
for t in pick:
    d = f"{train}/{t}"
    trk = io.load_track(d); nodes, arcs = io.load_network(d)
    net = RoadNetwork(nodes, arcs, LocalProjection.from_track(trk))
    route = io.load_route(d)
    t0 = time.time()
    lab = label_trip(d, trk, net)
    sec = time.time() - t0
    px, py = net.proj.to_xy(trk.lon, trk.lat)
    near = route[nearest_route_arc(px, py, route, net)]
    agree = (near == lab.true_arc_id.to_numpy()).mean()
    mono = bool((np.diff(lab.route_idx.to_numpy()) >= 0).all())
    dd = lab.label_dist_m
    print(f"{t}, {len(trk)}, {len(route)}, {sec:.1f}, {dd.mean():.1f}, {np.percentile(dd,95):.1f}, {mono}, {agree:.3f}")
    assert mono and len(lab) == len(trk)
print("ALL OK")
