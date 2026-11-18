"""Dataset inventory (EDA step 1). Run from the repo root:  python scripts/run_eda.py
Writes experiments/results/eda_inventory.csv  (one row per trip, train + student_test only)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from pathsnap import io
from pathsnap.geo import LocalProjection

RAW, RES = ROOT / "data" / "raw", ROOT / "experiments" / "results"
RES.mkdir(parents=True, exist_ok=True)
rows = []
for split in ["train", "student_test"]:
    for t in io.list_trips(RAW / split):
        d = RAW / split / t
        trk = io.load_track(d)
        nodes, arcs = io.load_network(d)
        route = io.load_route(d)                        # EDA on train/student_test only
        x, y = LocalProjection.from_track(trk).to_xy(trk["lon"], trk["lat"])
        dt = np.diff(trk["t"].to_numpy())
        rows.append(dict(trip_id=t, partition=split, n_points=len(trk), n_nodes=len(nodes), n_arcs=len(arcs),
                         route_len=len(route), size_kb=round(sum(f.stat().st_size for f in d.iterdir()) / 1024),
                         duration_s=float(trk["t"].iloc[-1] - trk["t"].iloc[0]),
                         pct_1s_steps=round(float((dt == 1).mean()) * 100, 1), max_gap_s=float(dt.max()),
                         n_nan=int(trk.isna().sum().sum() + nodes.isna().sum().sum()),
                         time_sorted=bool((dt > 0).all()),
                         extent_x_km=round((x.max() - x.min()) / 1000, 2), extent_y_km=round((y.max() - y.min()) / 1000, 2),
                         mean_lon=round(trk["lon"].mean(), 4), mean_lat=round(trk["lat"].mean(), 4)))
inv = pd.DataFrame(rows)
inv.to_csv(RES / "eda_inventory.csv", index=False)
print(inv.groupby("partition")[["n_points", "n_nodes", "n_arcs", "route_len"]].agg(["count", "min", "median", "max"]).T.to_string())
print("missing values (NaN):", int(inv.n_nan.sum()), "| trips with unsorted timestamps:", int((~inv.time_sorted).sum()),
      "| trips with a gap > 60 s:", int((inv.max_gap_s > 60).sum()))
print(inv.sort_values("max_gap_s", ascending=False)[["trip_id", "partition", "max_gap_s"]].head(6).to_string(index=False))
print("wrote", RES / "eda_inventory.csv")
