"""EDA checks + figures (partitions: train and student_test only). Run from the repo root:
    python scripts/run_eda_figures.py          (run build_dataset.py first)
Needs: pip install matplotlib
Writes  experiments/results/eda_network_connectivity.csv, eda_sampling_intervals.csv
        reports/figures/eda_*.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap import io
from pathsnap.geo import LocalProjection
from pathsnap.network import RoadNetwork

RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
RES, FIG = ROOT / "experiments" / "results", ROOT / "reports" / "figures"
RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
COL = {"train": "#2166ac", "student_test": "#d6604d"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---------- 1. per-trip checks: sampling, network connectivity, route structure ----------
rows, dts, centres = [], [], {}
for split in ["train", "student_test"]:
    for t in io.list_trips(RAW / split):
        d = RAW / split / t
        trk = io.load_track(d); nodes, arcs = io.load_network(d); route = io.load_route(d)
        dt = np.diff(trk["t"].to_numpy()); dts.append(dt)
        n, f, to = len(nodes), arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
        g = csr_matrix((np.ones(len(f)), (f, to)), shape=(n, n))
        nw, lw = connected_components(g, directed=True, connection="weak")
        ns, ls = connected_components(g, directed=True, connection="strong")
        key = f.astype(np.int64) * n + to
        rows.append(dict(trip_id=t, partition=split, n_nodes=n, n_arcs=len(arcs),
                         pct_arcs_with_reverse_twin=round(100 * np.isin(to.astype(np.int64) * n + f, key).mean(), 1),
                         isolated_nodes=int(((np.bincount(f, minlength=n) + np.bincount(to, minlength=n)) == 0).sum()),
                         weak_components=nw, largest_weak_share=round(np.bincount(lw).max() / n, 3),
                         strong_components=ns, largest_strong_share=round(np.bincount(ls).max() / n, 3),
                         route_len=len(route), route_connected_pct=round(100 * (to[route[:-1]] == f[route[1:]]).mean(), 1),
                         mean_lon=trk["lon"].mean(), mean_lat=trk["lat"].mean()))
net = pd.DataFrame(rows)
net.drop(columns=["mean_lon", "mean_lat"]).to_csv(RES / "eda_network_connectivity.csv", index=False)

alldt = np.concatenate(dts)
bins = [("1 s", alldt == 1), ("(1, 2] s", (alldt > 1) & (alldt <= 2)), ("(2, 10] s", (alldt > 2) & (alldt <= 10)),
        ("(10, 60] s", (alldt > 10) & (alldt <= 60)), ("> 60 s", alldt > 60)]
samp = pd.DataFrame({"interval": [b[0] for b in bins], "count": [int(b[1].sum()) for b in bins]})
samp["share_pct"] = (100 * samp["count"] / len(alldt)).round(3)
samp.to_csv(RES / "eda_sampling_intervals.csv", index=False)
print(samp.to_string(index=False))
print(net.groupby("partition")[["isolated_nodes", "weak_components", "strong_components"]].median().T.to_string())
print("largest strongly-connected share: median", net.largest_strong_share.median(), "min", net.largest_strong_share.min(),
      "| trips with >1 weak component:", int((net.weak_components > 1).sum()),
      "| route connected % (min):", net.route_connected_pct.min())

# ---------- 2. figures ----------
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.bar(samp["interval"], samp["share_pct"], color="#2166ac"); ax.set_yscale("log")
ax.set_ylabel("% of time steps (log)"); ax.set_title("GPS sampling interval (90 trips)")
fig.tight_layout(); fig.savefig(FIG / "eda_sampling_intervals.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.8, 2.8))
for s, g in net.groupby("partition"):
    ax.scatter(g.mean_lon, g.mean_lat, s=14, color=COL[s], label=f"{s} ({len(g)})", alpha=.85)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("Trip locations (mean GPS position)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "eda_trip_locations.png", dpi=200); plt.close(fig)

fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.7))
for s, g in net.groupby("partition"):
    a.scatter(g.n_arcs, g.n_nodes, s=14, color=COL[s], label=s)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("arcs"); a.set_ylabel("nodes"); a.set_title("Road-network size per trip"); a.legend(frameon=False)
b.hist(net.largest_strong_share, bins=20, color="#2166ac"); b.set_xlabel("share of nodes in largest strongly-connected component")
b.set_ylabel("trips"); b.set_title("Network connectivity")
fig.tight_layout(); fig.savefig(FIG / "eda_network_connectivity.png", dpi=200); plt.close(fig)

# GPS-to-reference distance (from the aligned labels built by build_dataset.py)
def read(name):
    p = PROC / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(PROC / f"{name}.csv.gz")
ol = pd.concat([read("train_obs_labels").assign(partition="train"), read("student_test_obs_labels").assign(partition="student_test")])
dist = ol["label_dist_m"].clip(upper=100)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.hist(dist, bins=50, color="#2166ac"); ax.set_yscale("log"); ax.axvline(50, color="#d6604d", ls="--", label="candidate radius 50 m")
ax.set_xlabel("distance GPS point -> its reference road (m; last bar = 100 m or more)"); ax.set_ylabel("GPS points (log)")
ax.set_title("GPS noise vs reference route"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "eda_label_distance.png", dpi=200); plt.close(fig)
print("GPS->reference distance: median %.1f m, 95th pct %.1f m, share > 50 m: %.3f" % (ol.label_dist_m.median(), ol.label_dist_m.quantile(.95), (ol.label_dist_m > 50).mean()))

def trip_axes(ax, split, t):
    d = RAW / split / t
    trk = io.load_track(d); nodes, arcs = io.load_network(d); route = io.load_route(d)
    n = RoadNetwork(nodes, arcs, LocalProjection.from_track(trk))
    px, py = n.proj.to_xy(trk["lon"], trk["lat"])
    _, near, _, _ = n.candidates_near(px[::5], py[::5], 100.0); near = np.unique(near)
    for a in near[:3000]:
        ax.plot([n.x1[a], n.x2[a]], [n.y1[a], n.y2[a]], color="#bbbbbb", lw=.6, zorder=1)
    for a in route:
        ax.plot([n.x1[a], n.x2[a]], [n.y1[a], n.y2[a]], color="#1a9850", lw=3.2, zorder=2)
    ax.scatter(px, py, s=1.2, color="#2166ac", zorder=3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"trip {t} ({split})", fontsize=8)

# small multiples of 6 trips
picks = [("student_test", t) for t in io.list_trips(RAW / "student_test")[:3]] + [("train", t) for t in io.list_trips(RAW / "train")[:3]]
fig, axs = plt.subplots(2, 3, figsize=(7.5, 5))
for ax, (s, t) in zip(axs.ravel(), picks): trip_axes(ax, s, t)
fig.suptitle("GPS (blue), reference route (green), nearby network (grey)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "eda_trajectories.png", dpi=200); plt.close(fig)

# 2-panel figure for the report: trajectory/network/route + GPS noise histogram
fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
trip_axes(a, "student_test", "00000052")
b.hist(dist, bins=50, color="#2166ac"); b.set_yscale("log"); b.axvline(50, color="#d6604d", ls="--", label="radius 50 m")
b.set_xlabel("GPS point -> reference road (m; last bar = 100 m or more)"); b.set_ylabel("GPS points (log)"); b.set_title("GPS noise vs reference", fontsize=8); b.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "eda_report_panel.png", dpi=250); plt.close(fig)
print("figures written to", FIG)
