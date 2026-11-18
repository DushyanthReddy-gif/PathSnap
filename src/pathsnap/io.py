"""Readers for the four raw PathSnap file types.

Folder layout (one folder per trip, trip id = 8-character string):
    <split_dir>/<trip_id>/<trip_id>.track   lon lat t      (GPS points)
    <split_dir>/<trip_id>/<trip_id>.nodes   lon lat        (row number = node_id)
    <split_dir>/<trip_id>/<trip_id>.arcs    from to        (row number = arc_id)
    <split_dir>/<trip_id>/<trip_id>.route   arc_id         (reference route, may be absent)

None of the files has a header, and the column order is LONGITUDE, then LATITUDE.

LEAKAGE RULE: load_route() may only be called from labels.py (training labels)
and evaluation/ (scoring). The prediction path must never call it.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def list_trips(split_dir):
    """Sorted trip ids (folder names) found inside split_dir, e.g. ['00000000', ...]."""
    split_dir = Path(split_dir)
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def _trip_file(trip_dir, ext):
    trip_dir = Path(trip_dir)
    return trip_dir / f"{trip_dir.name}.{ext}"


def _read(path, names, dtype):
    # whitespace-separated, no header
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def load_track(trip_dir):
    """GPS points as a DataFrame with columns lon, lat, t. Row index = obs_idx (0-based)."""
    df = _read(_trip_file(trip_dir, "track"), ["lon", "lat", "t"], "float64")
    df.index.name = "obs_idx"
    return df


def load_network(trip_dir):
    """Returns (nodes_df[lon, lat], arcs_df[from_node, to_node]).

    nodes_df row index = node_id, arcs_df row index = arc_id.
    """
    nodes = _read(_trip_file(trip_dir, "nodes"), ["lon", "lat"], "float64")
    arcs = _read(_trip_file(trip_dir, "arcs"), ["from_node", "to_node"], "int64")
    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"
    return nodes, arcs


def has_route(trip_dir):
    return _trip_file(trip_dir, "route").exists()


def load_route(trip_dir):
    """Ordered array of reference arc_ids. ONLY labels.py and evaluation/ may call this."""
    df = pd.read_csv(_trip_file(trip_dir, "route"), header=None, dtype="int64")
    return df[0].to_numpy()
