"""Candidate radius / K sweep on 40 train trips (every 2nd), trusted labels only. Run from repo root:
    python scripts/candidate_sweep.py        -> experiments/results/eda_candidate_sweep.csv"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from pathsnap import io
from pathsnap.geo import LocalProjection
from pathsnap.network import RoadNetwork
from pathsnap.candidates import generate_candidates
from pathsnap.labels import label_trip
R = str(ROOT / "data" / "raw" / "train")
rows = []
trips = io.list_trips(R)[::2]            # 40 train trips for speed
for t in trips:
    d = f"{R}/{t}"; trk = io.load_track(d); nodes, arcs = io.load_network(d)
    net = RoadNetwork(nodes, arcs, LocalProjection.from_track(trk))
    lab = label_trip(d, trk, net)
    ok = (lab.label_dist_m <= 50).to_numpy()
    sub = np.arange(0, len(trk), 3)
    c = generate_candidates(trk, net, 100.0, 1000, sub)           # widest setting, then filter
    c["hit"] = (c.arc_id.to_numpy() == lab.true_arc_id.to_numpy()[c.obs_idx.to_numpy()])
    c["rank"] = c.groupby("obs_idx").cumcount()
    for r in (20, 30, 50, 75, 100):
        for k in (3, 5, 8, 12):
            cc = c[(c.dist_m <= r) & (c["rank"] < k)]
            hit = cc.groupby("obs_idx")["hit"].max().reindex(sub, fill_value=False).to_numpy()
            rows.append((r, k, hit[ok[sub]].sum(), ok[sub].sum(), len(cc)))
df = pd.DataFrame(rows, columns=["radius_m", "K", "hits", "n_trusted_obs", "cand_rows"]).groupby(["radius_m", "K"]).sum().reset_index()
df["miss_rate"] = 1 - df.hits / df.n_trusted_obs
df["rows_per_obs"] = df.cand_rows / df.n_trusted_obs
print(df[["radius_m", "K", "miss_rate", "rows_per_obs"]].round(4).to_string(index=False))
df[["radius_m", "K", "miss_rate", "rows_per_obs"]].to_csv(str(ROOT / "experiments" / "results" / "eda_candidate_sweep.csv"), index=False)
