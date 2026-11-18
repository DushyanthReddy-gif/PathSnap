"""Tune the HMM decoder (alpha, sigma_m, beta) on TRAIN trips with out-of-fold probabilities.

Run from anywhere:
    python scripts/tune_hmm.py

Why train + out-of-fold: the guidelines allow decoder tuning on training and student-test data
only. Tuning on train keeps student-test clean for the final comparison, and out-of-fold
probabilities (each from a model that never saw that trip) match the confidence the deployed
model has on new trips; in-sample probabilities would be over-confident.

- Trips: a fixed random sample (seed 42) of 16 train trips whose reference labels are clean
  (<= 5% of points farther than 50 m from their reference road); trusted points only.
  Note: the train table keeps every 3rd GPS point, so steps are ~3 s instead of 1 s.
- Grid: alpha (classifier weight in the emission) x sigma_m (GPS noise) x beta (transition scale).
- Score per config: mean over trips of HMM Top-1 (selection metric), raw per-point route
  continuity (tie-break), and route F1 of the reconstructed route.
- Each trip runs in its own process and goes through the whole grid on one RoadNetwork.

Outputs:
    experiments/results/hmm_param_search.csv   one row per config x trip
    experiments/results/hmm_param_summary.csv  one row per config (means), best first
    artifacts/hmm_params.json                  best config (other decoder settings kept)
    experiments/logs/tune_hmm_<timestamp>.log
"""
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SIGMAS = [5.0, 10.0, 20.0]
BETAS = [5.0, 20.0, 50.0]
N_TRIPS, SEED = 16, 42
RES, LOGS = ROOT / "experiments" / "results", ROOT / "experiments" / "logs"


def run_trip(trip):
    """All grid configs on one trip. Returns a list of result rows."""
    from pathsnap import io
    from pathsnap.evaluation.route_metrics import route_continuity, sequence_metrics
    from pathsnap.geo import LocalProjection
    from pathsnap.hmm.route import reconstruct_route
    from pathsnap.hmm.viterbi import decode
    from pathsnap.network import RoadNetwork

    d = ROOT / "data" / "raw" / "train" / trip
    trk = io.load_track(d)
    nodes, arcs = io.load_network(d)
    ref = io.load_route(d)                                  # evaluation only (route F1)
    net = RoadNetwork(nodes, arcs, LocalProjection.from_track(trk))
    cands = pd.read_parquet(ROOT / "data" / "processed" / "train_candidates.parquet",
                            filters=[("trip_id", "==", trip)])
    oof = pd.read_parquet(RES / "probs" / "oof_train_hist_gb_tuned.parquet", filters=[("trip_id", "==", trip)])
    cands = cands[cands["label_ok"] == 1].merge(oof[["obs_idx", "arc_id", "p"]], on=["obs_idx", "arc_id"])
    labels = pd.read_parquet(ROOT / "data" / "processed" / "train_obs_labels.parquet", filters=[("trip_id", "==", trip)])
    labels = labels[labels["label_dist_m"] <= 50][["obs_idx", "true_arc_id"]]

    rows = []
    for alpha, sigma, beta in itertools.product(ALPHAS, SIGMAS, BETAS):
        params = {"alpha": alpha, "sigma_m": sigma, "beta": beta, "use_classifier_probabilities": alpha > 0}
        t0 = time.perf_counter()
        h = decode(cands, net, params)
        dt = time.perf_counter() - t0
        m = labels.merge(h, on="obs_idx", how="left")
        per_obs = [a for a in h.sort_values("obs_idx")["arc_id"] if a >= 0]
        route = reconstruct_route(per_obs, net)
        rows.append({"alpha": alpha, "sigma_m": sigma, "beta": beta, "trip_id": trip,
                     "n_obs": len(labels), "hmm_top1": float((m["arc_id"] == m["true_arc_id"]).mean()),
                     "continuity_raw": route_continuity(per_obs, net),
                     "route_f1": sequence_metrics(route, ref)["f1"], "decode_s": dt})
    return rows


if __name__ == "__main__":
    LOGS.mkdir(parents=True, exist_ok=True)
    log_f = open(LOGS / f"tune_hmm_{datetime.now():%Y%m%d_%H%M%S}.log", "w")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    q = pd.read_csv(RES / "eda_label_quality.csv", dtype={"trip_id": str})
    clean = q[(q["split"] == "train") & (q["share_label_gt_limit"] <= 0.05)]["trip_id"].str.zfill(8)
    trips = sorted(clean.sample(n=N_TRIPS, random_state=SEED).tolist())
    log(f"[setup] {len(clean)} clean train trips, tuning on {len(trips)}: {trips}")
    log(f"[setup] grid alpha={ALPHAS} sigma_m={SIGMAS} beta={BETAS} -> {len(ALPHAS) * len(SIGMAS) * len(BETAS)} configs")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=len(trips)) as ex:
        rows = [r for trip_rows in ex.map(run_trip, trips) for r in trip_rows]
    log(f"[search] done in {time.perf_counter() - t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "hmm_param_search.csv", index=False)
    summ = (df.groupby(["alpha", "sigma_m", "beta"])[["hmm_top1", "continuity_raw", "route_f1", "decode_s"]]
              .mean().reset_index().sort_values(["hmm_top1", "continuity_raw"], ascending=False))
    summ.to_csv(RES / "hmm_param_summary.csv", index=False)
    log("\n=== top 8 configs (mean over trips)\n" + summ.head(8).round(4).to_string(index=False))
    log("\n=== distance-only (alpha=0) best\n" + summ[summ.alpha == 0].head(1).round(4).to_string(index=False))

    best = summ.iloc[0]
    hp_path = ROOT / "artifacts" / "hmm_params.json"
    hp = json.load(open(hp_path))
    hp.update({"alpha": float(best.alpha), "sigma_m": float(best.sigma_m), "beta": float(best.beta),
               "use_classifier_probabilities": bool(best.alpha > 0)})
    json.dump(hp, open(hp_path, "w"), indent=2)
    log(f"\n[best] alpha={best.alpha} sigma_m={best.sigma_m} beta={best.beta} -> artifacts/hmm_params.json")
