"""Training / cross-validation harness (owner: P2 - Aditya).

Key ideas (for the viva):
- GroupKFold by trip_id: every GPS observation of a trip stays in the same fold, so each
  model is always validated on trips it has never seen. A random K-fold would put points
  of the same drive in train and validation and inflate the scores.
- Subsampling (max_train_obs) picks WHOLE observations (all their candidates), so the
  "one true arc per observation" structure and the Top-1 metric stay valid.
- For every fold we also score the training part -> train-vs-validation gap is the
  evidence for over/underfitting (report §3.2 Q9).
- Out-of-fold (OOF) probabilities: every training candidate gets a probability from the
  model that did NOT see its trip. P3 tunes the HMM on these, so the HMM is not tuned on
  over-confident in-sample probabilities.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.evaluation.candidate_metrics import evaluate
from pathsnap.models.registry import make_model

ID_COLS = ["trip_id", "obs_idx", "t", "arc_id", "label"]
OBS_KEY = ["trip_id", "obs_idx"]


# ---------------------------------------------------------------- data
def load_split(processed_dir, split):
    """Returns (candidates_df, obs_labels_df) for 'train' or 'student_test'."""
    d = Path(processed_dir)
    cands = pd.read_parquet(d / f"{split}_candidates.parquet")
    obs = pd.read_parquet(d / f"{split}_obs_labels.parquet")
    cands["trip_id"] = cands["trip_id"].astype(str)
    obs["trip_id"] = obs["trip_id"].astype(str)
    return cands, obs


def feature_columns(cands):
    """FEATURE_COLUMNS from P1's features.py if available, else every non-id numeric column."""
    try:
        from pathsnap.features import FEATURE_COLUMNS
        return list(FEATURE_COLUMNS)
    except Exception:
        return [c for c in cands.columns
                if c not in ID_COLS and pd.api.types.is_numeric_dtype(cands[c])]


def subsample_obs(cands, max_obs, seed):
    """Keep a random subset of max_obs observations (all candidates of each kept observation)."""
    if not max_obs:
        return cands
    keys = cands[OBS_KEY].drop_duplicates()
    if len(keys) <= max_obs:
        return cands
    keep = keys.sample(n=max_obs, random_state=seed)
    return cands.merge(keep, on=OBS_KEY, how="inner")


def _obs_for(obs_labels, cands):
    """Observation labels for exactly the observations present in `cands`
    plus candidate-less observations of the same trips (they must count as wrong)."""
    trips = cands["trip_id"].unique()
    return obs_labels[obs_labels["trip_id"].isin(trips)]


# ---------------------------------------------------------------- fit / score
def fit(cfg, cands, feats, seed=42, params_override=None):
    """Fit one model on a candidate table. Returns (fitted pipeline, fit seconds, rows used)."""
    tr = subsample_obs(cands, cfg.get("max_train_obs"), seed)
    model = make_model(cfg, seed=seed, params_override=params_override)
    t0 = time.perf_counter()
    model.fit(tr[feats].to_numpy(np.float32), tr["label"].to_numpy())
    return model, time.perf_counter() - t0, len(tr)


def score(model, cands, obs_labels, feats):
    """Metrics on one partition + inference time. Returns (metrics dict, probabilities)."""
    t0 = time.perf_counter()
    p = model.predict_proba(cands[feats].to_numpy(np.float32))[:, 1]
    pred_s = time.perf_counter() - t0
    m, _ = evaluate(cands, p, obs_labels)
    n_obs = cands[OBS_KEY].drop_duplicates().shape[0]
    m["inference_ms_per_1k_obs"] = 1000.0 * pred_s / max(n_obs, 1) * 1000.0
    return m, p


def cross_validate(name, cfg, cands, obs_labels, feats, n_splits=5, seed=42,
                   params_override=None, train_score_obs=20000, log=print):
    """GroupKFold CV on the training trips.

    Returns (fold_results DataFrame, oof probability array aligned with `cands` rows).
    """
    groups = cands["trip_id"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(cands), np.nan)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(cands, groups=groups)):
        tr, va = cands.iloc[tr_idx], cands.iloc[va_idx]
        model, fit_s, n_rows = fit(cfg, tr, feats, seed=seed, params_override=params_override)
        va_m, p = score(model, va, _obs_for(obs_labels, va), feats)
        oof[va_idx] = p
        # training-fold score on a subsample (over/underfitting evidence)
        tr_s = subsample_obs(tr, train_score_obs, seed)
        tr_m, _ = score(model, tr_s, obs_labels.merge(tr_s[OBS_KEY].drop_duplicates(), on=OBS_KEY), feats)
        rows.append({
            "model": name, "fold": fold,
            "n_train_trips": tr["trip_id"].nunique(), "n_val_trips": va["trip_id"].nunique(),
            "train_rows_used": n_rows, "fit_s": fit_s,
            "val_macro_f1": va_m["macro_f1"], "val_top1": va_m["top1_accuracy"],
            "val_log_loss": va_m["log_loss"], "val_precision": va_m["precision"],
            "val_recall": va_m["recall"], "val_f1": va_m["f1"], "val_accuracy": va_m["accuracy"],
            "train_macro_f1": tr_m["macro_f1"], "train_top1": tr_m["top1_accuracy"],
            "train_log_loss": tr_m["log_loss"],
            "inference_ms_per_1k_obs": va_m["inference_ms_per_1k_obs"],
        })
        log(f"  [{name}] fold {fold}: val macroF1={va_m['macro_f1']:.4f} top1={va_m['top1_accuracy']:.4f} "
            f"logloss={va_m['log_loss']:.4f} | train top1={tr_m['top1_accuracy']:.4f} | fit {fit_s:.1f}s")
    return pd.DataFrame(rows), oof
