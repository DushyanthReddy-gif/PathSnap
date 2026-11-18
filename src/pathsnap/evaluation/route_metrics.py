"""
Route-level evaluation metrics (P3 — Leka).

Public functions
----------------
candidate_miss_rate(obs_labels_df)                -> float
route_continuity(arc_sequence, network)            -> float
sequence_metrics(pred_arc_sequence, ref_route)     -> dict

Signatures must match §4.6 of 00_TEAM_PLAN.md exactly; both P2 and P4
import from here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── candidate_miss_rate ────────────────────────────────────────────────────

def candidate_miss_rate(obs_labels_df: pd.DataFrame) -> float:
    """
    Fraction of observations whose true arc was NOT in the candidate set.

    Parameters
    ----------
    obs_labels_df : DataFrame with columns  trip_id, obs_idx, t,
                    true_arc_id, n_candidates, true_in_candidates.
                    (§4.2 of the team plan.)

    Returns
    -------
    float in [0, 1].  0 means all true arcs were reachable as candidates.

    Definition
    ----------
    miss_rate = sum(true_in_candidates == False) / len(obs_labels_df)
    """
    if len(obs_labels_df) == 0:
        return 0.0
    missed = (~obs_labels_df["true_in_candidates"].astype(bool)).sum()
    return float(missed) / len(obs_labels_df)


# ── route_continuity ───────────────────────────────────────────────────────

def route_continuity(arc_sequence: "list[int] | np.ndarray", network) -> float:
    """
    Fraction of consecutive arc pairs that are directly connected in the
    road network (to_node(a) == from_node(b)), after collapsing runs of
    the same arc.

    Parameters
    ----------
    arc_sequence : ordered list / array of predicted arc_ids (one per obs).
                   May contain -1 for observations with no candidates.
    network      : RoadNetwork object with attributes arcs_df (or from_node /
                   to_node lookup arrays) for endpoint retrieval.

    Returns
    -------
    float in [0, 1].  1.0 means every consecutive pair is connected.
    Returns 1.0 for sequences of length ≤ 1 (trivially connected).

    Definition
    ----------
    After removing -1 entries and collapsing consecutive duplicate arcs,
    count pairs (a, b) where to_node(a) == from_node(b).
    continuity = connected_pairs / total_pairs
    """
    arcs = np.asarray(arc_sequence, dtype=np.int64)
    # Remove -1 (missing arc assignments)
    arcs = arcs[arcs >= 0]
    # Collapse consecutive duplicates
    if len(arcs) > 1:
        mask = np.concatenate([[True], arcs[1:] != arcs[:-1]])
        arcs = arcs[mask]

    if len(arcs) <= 1:
        return 1.0

    # Build endpoint lookup
    def _to_node(arc_id: int) -> int:
        try:
            return int(network.arcs_df.iloc[arc_id]["to_node"])
        except AttributeError:
            return int(network.to_node[arc_id])

    def _from_node(arc_id: int) -> int:
        try:
            return int(network.arcs_df.iloc[arc_id]["from_node"])
        except AttributeError:
            return int(network.from_node[arc_id])

    connected = 0
    total = len(arcs) - 1
    for i in range(total):
        a, b = int(arcs[i]), int(arcs[i + 1])
        try:
            if _to_node(a) == _from_node(b):
                connected += 1
        except (IndexError, KeyError):
            # Arc ID out of range — treat as disconnected
            pass

    return connected / total if total > 0 else 1.0


# ── sequence_metrics ───────────────────────────────────────────────────────

def sequence_metrics(
    pred_arc_sequence: "list[int] | np.ndarray",
    ref_route: "list[int] | np.ndarray",
) -> dict[str, float]:
    """
    Route-level precision, recall, and F1 over the *set* of arc_ids.

    Parameters
    ----------
    pred_arc_sequence : ordered list of predicted arc_ids (one per obs).
                        -1 entries are excluded.
    ref_route         : ordered list of arc_ids from the .route file (ground
                        truth for the whole trip).

    Returns
    -------
    dict with keys:
        "precision"   : |pred ∩ ref| / |pred|
        "recall"      : |pred ∩ ref| / |ref|
        "f1"          : harmonic mean of precision and recall
        "exact_match" : 1.0 iff the two sequences are identical

    Length-weighted variant (stretch)
    ----------------------------------
    If the network is passed as a third argument, lengths can be incorporated.
    That extension is documented in the report but not required by the interface.
    """
    pred = np.asarray(pred_arc_sequence, dtype=np.int64)
    ref  = np.asarray(ref_route, dtype=np.int64)

    # Remove -1
    pred = pred[pred >= 0]
    ref  = ref[ref >= 0]

    pred_set = set(pred.tolist())
    ref_set  = set(ref.tolist())

    intersection = len(pred_set & ref_set)

    precision = intersection / len(pred_set) if pred_set else 0.0
    recall    = intersection / len(ref_set)  if ref_set  else 0.0

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    exact = 1.0 if (len(pred) == len(ref) and np.array_equal(pred, ref)) else 0.0

    return {
        "precision":   round(precision, 6),
        "recall":      round(recall, 6),
        "f1":          round(f1, 6),
        "exact_match": exact,
    }
