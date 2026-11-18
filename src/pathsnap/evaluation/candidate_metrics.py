"""Candidate-level and observation-level metrics (owner: P2 - Aditya).

Two levels of evaluation:

1. Candidate level - every row is one (GPS observation, candidate arc) pair with
   label 1 if that arc is the reference arc for the observation, else 0.
   -> accuracy, precision, recall, F1, macro F1, log loss, confusion matrix.

2. Observation level - for each GPS observation pick the candidate with the
   highest predicted probability p.
   -> Top-1 accuracy: fraction of observations whose picked arc == reference arc.
      Observations whose reference arc is NOT among their candidates count as wrong
      (the model never had a chance; that is the candidate miss rate's job to show).

Decision rule used for the class predictions behind accuracy / P / R / F1:
   rule="top1"      (team default) a candidate is predicted 1 iff it has the highest p
                    inside its observation. Threshold-free, so models are compared
                    fairly whatever their calibration, and it is exactly the decision
                    the route reconstruction makes.
   rule="threshold" predicted 1 iff p >= threshold (kept for reference).
Log loss always uses the raw probabilities.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss,
                             precision_recall_fscore_support)

OBS_KEY = ["trip_id", "obs_idx"]
_EPS = 1e-7


def top1_mask(cands, p):
    """Boolean array, True for the highest-p candidate of each observation (ties -> first row)."""
    df = cands[OBS_KEY].copy()
    df["p"] = np.asarray(p, dtype="float64")
    df["_row"] = np.arange(len(df))
    best = (df.sort_values(OBS_KEY + ["p"], ascending=[True, True, False], kind="stable")
              .drop_duplicates(OBS_KEY)["_row"].to_numpy())
    mask = np.zeros(len(df), dtype=bool)
    mask[best] = True
    return mask


def top1_predictions(cands, p):
    """One row per observation that has candidates: trip_id, obs_idx, arc_id, p (the argmax)."""
    mask = top1_mask(cands, p)
    out = cands.loc[mask, OBS_KEY + ["arc_id"]].copy()
    out["p"] = np.asarray(p, dtype="float64")[mask]
    return out.reset_index(drop=True)


def top1_accuracy(pred_obs, obs_labels, covered_only=False):
    """Observation-level Top-1 accuracy.

    pred_obs   : DataFrame[trip_id, obs_idx, arc_id]  (one row per predicted observation)
    obs_labels : DataFrame[trip_id, obs_idx, true_arc_id, true_in_candidates] (ALL observations)
    covered_only=True restricts to observations whose true arc is among the candidates.
    """
    m = obs_labels[OBS_KEY + ["true_arc_id", "true_in_candidates"]].merge(
        pred_obs[OBS_KEY + ["arc_id"]], on=OBS_KEY, how="left")
    if covered_only:
        m = m[m["true_in_candidates"].astype(bool)]
    if len(m) == 0:
        return float("nan")
    return float((m["arc_id"] == m["true_arc_id"]).mean())     # NaN (no prediction) -> wrong


def candidate_report(y_true, p, cands=None, rule="top1", threshold=0.5):
    """Candidate-level classification metrics. `cands` (with trip_id, obs_idx) is needed for rule='top1'."""
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype="float64"), _EPS, 1 - _EPS)
    if rule == "top1":
        if cands is None:
            raise ValueError("rule='top1' needs the candidate table (trip_id, obs_idx)")
        y_pred = top1_mask(cands, p).astype(int)
    elif rule == "threshold":
        y_pred = (p >= threshold).astype(int)
    else:
        raise ValueError(f"unknown rule {rule!r}")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "rule": rule,
        "n_candidates": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(prec[1]),          # positive class (= "this is the true arc")
        "recall": float(rec[1]),
        "f1": float(f1[1]),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),  # [[TN, FP], [FN, TP]]
    }


def evaluate(cands, p, obs_labels, rule="top1"):
    """Everything for one data partition. Returns (metrics dict, top-1 predictions per observation)."""
    rep = candidate_report(cands["label"], p, cands, rule=rule)
    pred = top1_predictions(cands, p)
    rep["top1_accuracy"] = top1_accuracy(pred, obs_labels)
    rep["top1_accuracy_covered"] = top1_accuracy(pred, obs_labels, covered_only=True)
    rep["n_observations"] = int(len(obs_labels))
    return rep, pred
