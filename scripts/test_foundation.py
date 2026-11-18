"""Sanity tests for io / geo / network. Usage: python -I test_foundation.py <data_root>"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from pathsnap import io
from pathsnap.geo import LocalProjection, point_to_segment
from pathsnap.network import RoadNetwork

root = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "data" / "raw")  # default: repo data/raw
train = f"{root}/train"
trips = io.list_trips(train)
print("trips:", len(trips), trips[:3])

# find smallest and largest network
sizes = {t: sum(1 for _ in open(f"{train}/{t}/{t}.arcs")) for t in trips}
small, big = min(sizes, key=sizes.get), max(sizes, key=sizes.get)
print("smallest", small, sizes[small], "arcs | largest", big, sizes[big], "arcs")

for name, t in [("small", small), ("big", big)]:
    d = f"{train}/{t}"
    trk = io.load_track(d); nodes, arcs = io.load_network(d)
    t0 = time.time()
    proj = LocalProjection.from_track(trk)
    net = RoadNetwork(nodes, arcs, proj)
    t_build = time.time() - t0
    px, py = proj.to_xy(trk.lon, trk.lat)
    t0 = time.time()
    o, a, dist, fr = net.candidates_near(px, py, 50.0)
    t_q = time.time() - t0
    print(f"[{name}] {t}: obs={len(trk)} nodes={net.n_nodes} arcs={net.n_arcs} "
          f"build={t_build:.2f}s query={t_q:.2f}s pairs={len(o)}")
    cnt = np.bincount(o, minlength=len(trk))
    print(f"   candidates/obs: mean={cnt.mean():.1f} max={cnt.max()} obs with 0 cands={(cnt==0).sum()}")
    print(f"   length_m: min={net.length_m.min():.1f} median={np.median(net.length_m):.1f} max={net.length_m.max():.0f}")

    # brute-force check on 30 random observations
    rng = np.random.default_rng(0)
    bad = 0
    for i in rng.choice(len(trk), 30, replace=False):
        d_all, f_all = point_to_segment(px[i], py[i], net.x1, net.y1, net.x2, net.y2)
        truth = set(np.where(d_all <= 50.0)[0])
        got = set(a[o == i])
        bad += truth != got
        # distances must match too
        for arc_id, dd in zip(a[o == i], dist[o == i]):
            assert abs(dd - d_all[arc_id]) < 1e-6
    print(f"   brute-force mismatches (of 30 obs): {bad}")

    # projection round-trip + coordinate order sanity
    lon, lat = proj.to_lonlat(px, py)
    assert np.allclose(lon, trk.lon) and np.allclose(lat, trk.lat)
    # adjacency sanity: every out-arc starts at its node
    n = net.n_nodes // 2
    assert all(net.from_node[net.out_arcs[n]] == n) and all(net.to_node[net.in_arcs[n]] == n)

# a projection sanity check in metres: 0.001 deg lat ~ 111 m
p = LocalProjection(10.0, 60.0)
x, y = p.to_xy([10.0, 10.001], [60.0, 60.0]); print("0.001 deg lon at lat 60 =", round(float(x[1]-x[0]),1), "m (expect ~55.7)")
x, y = p.to_xy([10.0, 10.0], [60.0, 60.001]); print("0.001 deg lat =", round(float(y[1]-y[0]),1), "m (expect ~111.2)")

# a no-route folder must still load track/network
print("has_route(train trip):", io.has_route(f"{train}/{small}"))
print("route sample:", io.load_route(f"{train}/{small}")[:5])
print("ALL OK")
