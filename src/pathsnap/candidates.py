"""Candidate generation: for every GPS point, which roads could the car be on?

A candidate is a road (arc) passing within `radius_m` metres of the GPS point.  If more
than `max_k` roads qualify, only the `max_k` NEAREST are kept.

This module never reads the reference route, so it can run on hidden evaluator trips.
"""
import numpy as np
import pandas as pd

DEFAULT_RADIUS_M = 50.0
DEFAULT_MAX_K = 8


def generate_candidates(track, network, radius_m=DEFAULT_RADIUS_M, max_k=DEFAULT_MAX_K, obs_subset=None):
    """Returns a DataFrame with one row per (observation, candidate arc):
        obs_idx, t, arc_id, dist_m, proj_frac
    obs_subset: optional array of obs_idx to generate candidates for (default: all).
    """
    px, py = network.proj.to_xy(track["lon"].to_numpy(), track["lat"].to_numpy())
    idx = np.arange(len(track)) if obs_subset is None else np.asarray(obs_subset)
    o, a, d, f = network.candidates_near(px[idx], py[idx], radius_m)
    df = pd.DataFrame({"obs_idx": idx[o].astype("int32"), "arc_id": a.astype("int32"),
                       "dist_m": d.astype("float32"), "proj_frac": f.astype("float32")})
    df = df.sort_values(["obs_idx", "dist_m"], kind="stable")
    df = df[df.groupby("obs_idx").cumcount() < max_k].reset_index(drop=True)
    df.insert(1, "t", track["t"].to_numpy()[df["obs_idx"].to_numpy()].astype("float32"))
    return df
