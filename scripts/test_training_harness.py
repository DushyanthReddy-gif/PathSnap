"""End-to-end test of the training harness on a synthetic candidate table (agreed schema).
Usage (from repo root): python scripts/test_training_harness.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from pathsnap.models.registry import load_model_configs
from pathsnap.models.train import cross_validate

rng = np.random.default_rng(42)
rows, obs_rows = [], []
for trip in [f"{i:08d}" for i in range(10)]:
    noise = rng.uniform(5, 15)                                    # trips differ in GPS quality
    for obs in range(200):
        k = rng.integers(2, 9)
        covered = rng.random() > 0.08
        true_k = rng.integers(k)
        d = rng.exponential(25, k)
        if covered:
            d[true_k] = abs(rng.normal(0, noise))
        head = rng.uniform(0, 180, k)
        if covered:
            head[true_k] = abs(rng.normal(0, 20))
        rank = d.argsort().argsort()
        for j in range(k):
            rows.append((trip, obs, float(obs), int(rng.integers(1e5)), d[j], head[j] if rng.random() > 0.05 else np.nan,
                         rank[j], d[j] - d.min(), rng.random(), int(covered and j == true_k)))
        obs_rows.append((trip, obs, -1, covered))
cands = pd.DataFrame(rows, columns=["trip_id", "obs_idx", "t", "arc_id", "dist_m", "heading_diff",
                                    "dist_rank", "dist_minus_min", "proj_frac", "label"])
obs_labels = pd.DataFrame(obs_rows, columns=["trip_id", "obs_idx", "true_arc_id", "true_in_candidates"])
true_arc = cands[cands.label == 1].set_index(["trip_id", "obs_idx"])["arc_id"]
obs_labels = obs_labels.set_index(["trip_id", "obs_idx"])
obs_labels.loc[true_arc.index, "true_arc_id"] = true_arc
obs_labels = obs_labels.reset_index()
feats = ["dist_m", "heading_diff", "dist_rank", "dist_minus_min", "proj_frac"]
print(f"synthetic: {len(cands)} candidates, {len(obs_labels)} obs, {cands.trip_id.nunique()} trips, "
      f"NaNs in heading_diff={cands.heading_diff.isna().sum()}")

# 1. GroupKFold never splits a trip across train/validation
for tr, va in GroupKFold(5, shuffle=True, random_state=42).split(cands, groups=cands.trip_id):
    assert not set(cands.trip_id.iloc[tr]) & set(cands.trip_id.iloc[va])
print("GroupKFold: no trip leaks between train and validation folds")

cfgs = load_model_configs()["models"]
for name in ["logreg", "gaussian_nb", "lightgbm", "mlp"]:
    folds, oof = cross_validate(name, cfgs[name], cands, obs_labels, feats, n_splits=5, log=lambda *_: None)
    assert not np.isnan(oof).any(), "every row must get exactly one OOF probability"
    chance = obs_labels.true_in_candidates.mean() / cands.groupby(["trip_id", "obs_idx"]).size().mean()
    assert folds.val_top1.mean() > 2 * chance
    print(f"[{name:12s}] CV macroF1={folds.val_macro_f1.mean():.4f}±{folds.val_macro_f1.std():.4f} "
          f"top1={folds.val_top1.mean():.4f} (chance≈{chance:.3f}) logloss={folds.val_log_loss.mean():.4f} "
          f"train-top1={folds.train_top1.mean():.4f} fit={folds.fit_s.mean():.2f}s/fold "
          f"infer={folds.inference_ms_per_1k_obs.mean():.1f}ms/1k obs")
print("ALL OK")
