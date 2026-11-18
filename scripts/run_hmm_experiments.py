"""
run_hmm_experiments.py  —  P3 (Leka) experiment script.

Reproduces experiments/results/hmm_comparison.csv when run from the repo root.

Usage
-----
    python scripts/run_hmm_experiments.py [--split student_test] [--trips all]
                                          [--params artifacts/hmm_params.json]

What it does
------------
1. Loads student_test candidates (processed parquet or raw trip files).
2. For each trip, runs:
   a. Classifier-only prediction (argmax p per obs_idx).
   b. HMM decode() (using params from hmm_params.json).
3. Computes per-trip metrics: Top-1 accuracy, route continuity, sequence P/R/F1,
   decode time.
4. Writes results to experiments/results/hmm_comparison.csv.
5. Optionally runs a small grid search and writes hmm_param_search.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# ── make sure src/ is on the path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from pathsnap import io as pio
from pathsnap.geo import LocalProjection
from pathsnap.hmm.route import reconstruct_route
from pathsnap.hmm.viterbi import decode
from pathsnap.network import RoadNetwork
from pathsnap.evaluation.route_metrics import (
    candidate_miss_rate,
    route_continuity,
    sequence_metrics,
)


# ── helpers ───────────────────────────────────────────────────────────────

def _load_params(path: str) -> dict:
    with open(path) as f:
        p = json.load(f)
    # Remove metadata key
    p.pop("_note", None)
    return p


def _load_network(trip_dir: Path):
    """Load a minimal RoadNetwork-like object from raw files."""
    nodes = pd.read_csv(
        trip_dir / f"{trip_dir.name}.nodes", header=None, names=["lon", "lat"], sep=r"\s+"
    )
    arcs = pd.read_csv(
        trip_dir / f"{trip_dir.name}.arcs", header=None, names=["from_node", "to_node"], sep=r"\s+"
    )

    # Compute arc lengths in metres (rough — project degrees to metres locally)
    lat0 = nodes["lat"].mean()
    lon0 = nodes["lon"].mean()
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_132.0 * abs(float(np.cos(np.radians(lat0))))

    fn = arcs["from_node"].to_numpy()
    tn = arcs["to_node"].to_numpy()
    fn_lat = nodes.iloc[fn]["lat"].to_numpy()
    fn_lon = nodes.iloc[fn]["lon"].to_numpy()
    tn_lat = nodes.iloc[tn]["lat"].to_numpy()
    tn_lon = nodes.iloc[tn]["lon"].to_numpy()

    arcs["length_m"] = np.sqrt(
        ((tn_lat - fn_lat) * m_per_deg_lat) ** 2
        + ((tn_lon - fn_lon) * m_per_deg_lon) ** 2
    ).astype(np.float32)

    # Build adjacency list
    out_arcs: dict[int, list[int]] = {}
    for arc_id, row in arcs.iterrows():
        fn_node = int(row["from_node"])
        out_arcs.setdefault(fn_node, []).append(int(arc_id))

    class _Net:
        def __init__(self):
            self.arcs_df = arcs
            self.nodes_df = nodes
            self.out_arcs = out_arcs

    return _Net()


def _load_candidates(trip_id: str, processed_dir: Path, raw_dir: Path, split: str):
    """Load candidate table for a trip from parquet or fall back to raw."""
    parquet = processed_dir / f"{split}_candidates.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
        return df[df["trip_id"] == trip_id].copy()
    # Fallback: no processed data — return empty stub so the script doesn't crash
    print(f"  [warn] No processed parquet found, skipping {trip_id}")
    return pd.DataFrame()


def _load_route(trip_dir: Path) -> np.ndarray:
    tid = trip_dir.name
    route_file = trip_dir / f"{tid}.route"
    return np.loadtxt(str(route_file), dtype=np.int32)


def _classifier_only(cands: pd.DataFrame) -> pd.DataFrame:
    """Return one row per obs_idx: arc_id = argmax p (or first candidate)."""
    if "p" in cands.columns:
        idx = cands.groupby("obs_idx")["p"].idxmax()
        result = cands.loc[idx, ["obs_idx", "arc_id"]].copy()
    else:
        result = cands.groupby("obs_idx").first().reset_index()[["obs_idx", "arc_id"]].copy()
    result["obs_idx"] = result["obs_idx"].astype(np.int32)
    result["arc_id"] = result["arc_id"].astype(np.int32)
    return result.sort_values("obs_idx").reset_index(drop=True)


def _top1_accuracy(pred_df: pd.DataFrame, obs_labels: pd.DataFrame) -> float:
    """Fraction of observations where predicted arc == true arc."""
    merged = pred_df.merge(obs_labels[["obs_idx", "true_arc_id"]], on="obs_idx", how="inner")
    if len(merged) == 0:
        return 0.0
    return float((merged["arc_id"] == merged["true_arc_id"]).mean())


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run HMM experiments for all student_test trips.")
    parser.add_argument("--split", default="student_test")
    parser.add_argument("--trips", default="all", help="comma-separated trip IDs or 'all'")
    parser.add_argument("--params", default="artifacts/hmm_params.json")
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--probs", default="experiments/results/probs/student_test_hist_gb_tuned.parquet",
                        help="classifier probabilities (column p) of the deployed model for this split")
    args = parser.parse_args()
    probs = pd.read_parquet(ROOT / args.probs)
    probs["trip_id"] = probs["trip_id"].astype(str)
    print(f"Classifier probabilities: {args.probs} ({len(probs)} rows)")

    root = ROOT
    raw_dir = root / "data" / "raw" / args.split
    processed_dir = root / "data" / "processed"
    results_dir = root / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    params = _load_params(root / args.params)
    print("Loaded params:", params)

    # Discover trips
    if args.trips == "all":
        trip_dirs = sorted(raw_dir.iterdir()) if raw_dir.exists() else []
        trip_ids = [d.name for d in trip_dirs if d.is_dir()]
    else:
        trip_ids = [t.strip() for t in args.trips.split(",")]

    if not trip_ids:
        print(f"No trips found in {raw_dir}. Make sure data/raw/{args.split}/ is populated.")
        sys.exit(0)

    comparison_rows = []

    for trip_id in trip_ids:
        trip_dir = raw_dir / trip_id
        print(f"\n── Trip {trip_id} ──")

        cands = _load_candidates(trip_id, processed_dir, raw_dir, args.split)
        if cands.empty:
            continue
        # classifier probability p of the deployed model, per (obs, candidate arc)
        cands = cands.merge(probs[probs["trip_id"] == trip_id][["obs_idx", "arc_id", "p"]],
                            on=["obs_idx", "arc_id"], how="left")

        # same RoadNetwork (projection, lengths, adjacency) as the API pipeline
        _trk = pio.load_track(trip_dir)
        _nodes, _arcs = pio.load_network(trip_dir)
        network = RoadNetwork(_nodes, _arcs, LocalProjection.from_track(_trk))

        # Try to load labels (may not exist for inference-only split)
        labels_path = processed_dir / f"{args.split}_obs_labels.parquet"
        if labels_path.exists():
            obs_labels = pd.read_parquet(labels_path)
            obs_labels = obs_labels[obs_labels["trip_id"] == trip_id]
            has_labels = True
        else:
            obs_labels = pd.DataFrame()
            has_labels = False

        # Reference route
        try:
            ref_route = _load_route(trip_dir)
        except FileNotFoundError:
            ref_route = np.array([], dtype=np.int32)

        # ── Classifier-only ───────────────────────────────────────────────
        t0 = time.perf_counter()
        clf_pred = _classifier_only(cands)
        clf_time = time.perf_counter() - t0

        clf_top1 = _top1_accuracy(clf_pred, obs_labels) if has_labels else float("nan")
        clf_cont_raw = route_continuity(clf_pred["arc_id"].tolist(), network)        # per-point arcs
        clf_route = reconstruct_route(clf_pred["arc_id"].tolist(), network, params.get("cutoff_m", 2000.0))
        clf_cont = route_continuity(clf_route, network)                               # drivable route
        clf_seq  = sequence_metrics(clf_route, ref_route.tolist())

        # ── HMM decode ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        hmm_pred = decode(cands, network, params)
        hmm_time = time.perf_counter() - t0

        hmm_top1 = _top1_accuracy(hmm_pred, obs_labels) if has_labels else float("nan")
        hmm_arcs = hmm_pred.sort_values("obs_idx")["arc_id"].tolist()
        hmm_cont_raw = route_continuity(hmm_arcs, network)
        hmm_route = reconstruct_route(hmm_arcs, network, params.get("cutoff_m", 2000.0))
        hmm_cont = route_continuity(hmm_route, network)
        hmm_seq  = sequence_metrics(hmm_route, ref_route.tolist())

        print(
            f"  CLF  top1={clf_top1:.3f}  cont={clf_cont:.3f}  "
            f"f1={clf_seq['f1']:.3f}  t={clf_time:.2f}s"
        )
        print(
            f"  HMM  top1={hmm_top1:.3f}  cont={hmm_cont:.3f}  "
            f"f1={hmm_seq['f1']:.3f}  t={hmm_time:.2f}s"
        )

        comparison_rows.append({
            "trip_id": trip_id,
            "n_obs": len(clf_pred),
            # classifier-only
            "clf_top1": clf_top1,
            "clf_continuity_raw": clf_cont_raw,
            "clf_continuity": clf_cont,
            "clf_precision": clf_seq["precision"],
            "clf_recall": clf_seq["recall"],
            "clf_f1": clf_seq["f1"],
            "clf_time_s": round(clf_time, 4),
            # HMM
            "hmm_top1": hmm_top1,
            "hmm_continuity_raw": hmm_cont_raw,
            "hmm_continuity": hmm_cont,
            "hmm_precision": hmm_seq["precision"],
            "hmm_recall": hmm_seq["recall"],
            "hmm_f1": hmm_seq["f1"],
            "hmm_time_s": round(hmm_time, 4),
        })

    if not comparison_rows:
        print("No results to write.")
        return

    out_csv = results_dir / "hmm_comparison.csv"
    fieldnames = list(comparison_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    # Overall summary row
    df = pd.DataFrame(comparison_rows)
    print("\n── Overall averages ──")
    for col in ["clf_top1", "clf_continuity_raw", "clf_continuity", "clf_f1",
                "hmm_top1", "hmm_continuity_raw", "hmm_continuity", "hmm_f1", "hmm_time_s"]:
        print(f"  {col}: {df[col].mean():.4f}")

    print(f"\nResults written to {out_csv}")


if __name__ == "__main__":
    main()
