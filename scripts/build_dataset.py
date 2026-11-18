"""Build the model-ready tables from raw data.  Run from the repo root:

    python scripts/build_dataset.py

Writes (to data/processed/, which is git-ignored):
    {split}_candidates.parquet   one row per (GPS point, candidate road) + features + label
    {split}_obs_labels.parquet   one row per GPS point: true road, how many candidates, true road among them?
and experiments/results/eda_label_quality.csv (how well the reference route fits each trip's GPS).
If pyarrow is not installed it falls back to .csv.gz files.
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from pathsnap import io
from pathsnap.candidates import generate_candidates
from pathsnap.features import FEATURE_COLUMNS, build_features
from pathsnap.geo import LocalProjection
from pathsnap.labels import label_trip
from pathsnap.network import RoadNetwork

cfg = yaml.safe_load(open(ROOT / "configs" / "candidates.yaml"))
RAW, OUT, RES = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)


def save(df, name):
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(OUT / f"{name}.parquet", index=False)
    except ImportError:
        df.to_csv(OUT / f"{name}.csv.gz", index=False)


def process_trip(split, trip_id):
    d = RAW / split / trip_id
    trk = io.load_track(d)
    nodes, arcs = io.load_network(d)
    net = RoadNetwork(nodes, arcs, LocalProjection.from_track(trk))
    stride = cfg["train_obs_stride"] if split == "train" else 1
    subset = np.arange(0, len(trk), stride)

    c = build_features(generate_candidates(trk, net, cfg["radius_m"], cfg["max_k"], subset), trk, net)
    lab = label_trip(d, trk, net)                                   # training/evaluation only
    o = c["obs_idx"].to_numpy()
    c["label"] = (c["arc_id"].to_numpy() == lab["true_arc_id"].to_numpy()[o]).astype("int8")
    c["label_ok"] = (lab["label_dist_m"].to_numpy()[o] <= cfg["label_ok_max_dist_m"]).astype("int8")
    c.insert(0, "trip_id", trip_id)

    n_c = c.groupby("obs_idx").size().reindex(subset, fill_value=0)
    hit = c.groupby("obs_idx")["label"].max().reindex(subset, fill_value=0).astype(bool)
    ol = pd.DataFrame({"trip_id": trip_id, "obs_idx": subset.astype("int32"),
                       "t": trk["t"].to_numpy()[subset].astype("float32"),
                       "true_arc_id": lab["true_arc_id"].to_numpy()[subset],
                       "label_dist_m": lab["label_dist_m"].to_numpy()[subset],
                       "n_candidates": n_c.to_numpy().astype("int16"),
                       "true_in_candidates": hit.to_numpy()})
    q = lab["label_dist_m"].to_numpy()
    quality = dict(split=split, trip_id=trip_id, n_obs=len(trk), mean_label_dist_m=q.mean(),
                   p95_label_dist_m=np.percentile(q, 95), share_label_gt_limit=(q > cfg["label_ok_max_dist_m"]).mean())
    return c, ol, quality


t_start = time.time()
quality_rows = []
for split in ["train", "student_test"]:
    cs, ols = [], []
    for k, trip_id in enumerate(io.list_trips(RAW / split)):
        c, ol, q = process_trip(split, trip_id)
        cs.append(c); ols.append(ol); quality_rows.append(q)
    cand, obs = pd.concat(cs, ignore_index=True), pd.concat(ols, ignore_index=True)
    save(cand, f"{split}_candidates"); save(obs, f"{split}_obs_labels")
    ok = obs[obs["label_dist_m"] <= cfg["label_ok_max_dist_m"]]
    print(f"[{split}] trips={obs.trip_id.nunique()} obs={len(obs)} candidate_rows={len(cand)} "
          f"candidate_miss_rate={1 - obs.true_in_candidates.mean():.4f} "
          f"(on trusted labels only: {1 - ok.true_in_candidates.mean():.4f}; obs with 0 candidates: {(obs.n_candidates == 0).mean():.4f})")
pd.DataFrame(quality_rows).to_csv(RES / "eda_label_quality.csv", index=False)
print("FEATURE_COLUMNS =", FEATURE_COLUMNS)
print(f"done in {time.time() - t_start:.0f}s -> {OUT}")
