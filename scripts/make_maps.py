"""
make_maps.py  —  P3 (Leka) map generation script.

Usage
-----
    python scripts/make_maps.py --trip 00000052
    python scripts/make_maps.py --all

Produces Folium HTML files in reports/maps/.

Each map has 4 layers (§6 of P3_Leka_HMM_Viterbi_Maps.md):
  1. GPS points
  2. Road network context (arcs near the trajectory, capped at 3,000)
  3. Reference route (green)   — only produced by this script, never sent to UI
  4. Predicted HMM route (red)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from pathsnap.viz.maps import make_trip_map, build_arc_segments


# ── load helpers ──────────────────────────────────────────────────────────

def _load_track(trip_dir: Path) -> pd.DataFrame:
    tid = trip_dir.name
    df = pd.read_csv(
        trip_dir / f"{tid}.track", header=None, names=["lon", "lat", "t"], sep=r"\s+"
    )
    return df


def _load_nodes_arcs(trip_dir: Path):
    tid = trip_dir.name
    nodes = pd.read_csv(
        trip_dir / f"{tid}.nodes", header=None, names=["lon", "lat"], sep=r"\s+"
    )
    arcs = pd.read_csv(
        trip_dir / f"{tid}.arcs", header=None, names=["from_node", "to_node"], sep=r"\s+"
    )
    return nodes, arcs


def _load_route(trip_dir: Path) -> np.ndarray:
    tid = trip_dir.name
    return np.loadtxt(str(trip_dir / f"{tid}.route"), dtype=np.int32)


def _load_hmm_prediction(trip_id: str) -> pd.DataFrame | None:
    """Try to load a saved HMM prediction parquet for this trip."""
    p = ROOT / "experiments" / "results" / f"hmm_pred_{trip_id}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def _load_params() -> dict:
    p = ROOT / "artifacts" / "hmm_params.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        d.pop("_note", None)
        return d
    return {}


def _near_arcs(track_df: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame, radius_deg=0.01):
    """Return arc IDs whose midpoint is within radius_deg of any GPS point."""
    lat0 = track_df["lat"].mean()
    lon0 = track_df["lon"].mean()
    lat_r = track_df["lat"].max() - track_df["lat"].min() + radius_deg
    lon_r = track_df["lon"].max() - track_df["lon"].min() + radius_deg

    fn = arcs["from_node"].to_numpy()
    tn = arcs["to_node"].to_numpy()
    fn_lat = nodes.iloc[fn]["lat"].to_numpy()
    fn_lon = nodes.iloc[fn]["lon"].to_numpy()
    tn_lat = nodes.iloc[tn]["lat"].to_numpy()
    tn_lon = nodes.iloc[tn]["lon"].to_numpy()

    mid_lat = (fn_lat + tn_lat) / 2
    mid_lon = (fn_lon + tn_lon) / 2

    lat_min = track_df["lat"].min() - radius_deg
    lat_max = track_df["lat"].max() + radius_deg
    lon_min = track_df["lon"].min() - radius_deg
    lon_max = track_df["lon"].max() + radius_deg

    mask = (
        (mid_lat >= lat_min) & (mid_lat <= lat_max)
        & (mid_lon >= lon_min) & (mid_lon <= lon_max)
    )
    return np.where(mask)[0].tolist()


# ── make one map ──────────────────────────────────────────────────────────

def make_map_for_trip(trip_id: str, split: str = "student_test") -> None:
    trip_dir = ROOT / "data" / "raw" / split / trip_id
    if not trip_dir.exists():
        # Try train
        trip_dir = ROOT / "data" / "raw" / "train" / trip_id
    if not trip_dir.exists():
        print(f"[warn] Trip directory not found: {trip_id}")
        return

    print(f"Building map for trip {trip_id} ...")

    track = _load_track(trip_dir)
    nodes, arcs = _load_nodes_arcs(trip_dir)

    # GPS points as [lat, lon]  (raw files are lon, lat)
    gps_latlon = [[float(r.lat), float(r.lon)] for r in track.itertuples()]

    # Network context (nearby arcs, capped at 3,000)
    near = _near_arcs(track, nodes, arcs)[:3_000]
    network_segs = build_arc_segments(near, nodes, arcs, max_segs=3_000)

    # Reference route
    try:
        ref_route = _load_route(trip_dir)
        ref_segs = build_arc_segments(ref_route.tolist(), nodes, arcs)
    except FileNotFoundError:
        ref_segs = None

    # HMM prediction
    pred_df = _load_hmm_prediction(trip_id)
    if pred_df is not None:
        predicted_arc_ids = pred_df.sort_values("obs_idx")["arc_id"].tolist()
        pred_segs = build_arc_segments(predicted_arc_ids, nodes, arcs)
    else:
        pred_segs = None

    m = make_trip_map(
        gps_latlon=gps_latlon,
        network_segments=network_segs,
        reference_segments=ref_segs,
        predicted_segments=pred_segs,
        title=f"PathSnap — Trip {trip_id}",
    )

    out_dir = ROOT / "reports" / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trip_{trip_id}.html"
    m.save(str(out_path))
    print(f"  Saved → {out_path}")


# ── entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Folium maps for PathSnap trips.")
    parser.add_argument("--trip", help="single trip ID, e.g. 00000052")
    parser.add_argument("--all", action="store_true", help="generate maps for all student_test trips")
    parser.add_argument("--split", default="student_test")
    args = parser.parse_args()

    if args.trip:
        make_map_for_trip(args.trip, split=args.split)
    elif args.all:
        raw_dir = ROOT / "data" / "raw" / args.split
        trip_dirs = sorted(raw_dir.iterdir()) if raw_dir.exists() else []
        for d in trip_dirs:
            if d.is_dir():
                make_map_for_trip(d.name, split=args.split)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
