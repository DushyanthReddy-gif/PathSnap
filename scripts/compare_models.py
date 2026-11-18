"""Compare the 5 models of configs/models.yaml (report §3).

Run from anywhere:
    python scripts/compare_models.py                       # all 5 models
    python scripts/compare_models.py --models hist_gb      # one model

For every model:
  1. 5-fold GroupKFold cross-validation on the 80 train trips (trusted labels only)
     -> CV macro-F1 mean/std, CV Top-1, CV log loss, train-fold scores (overfitting evidence)
  2. refit on all 80 train trips, score the 10 student-test trips (EVERY observation)
     -> Top-1, macro-F1, precision, recall, F1, log loss, confusion matrix, inference time
  3. classifier-only route continuity on student_test (P3's route_continuity, same definition
     as the HMM comparison) from the top-1 arc of every observation in time order

Outputs (experiments/results/):
    model_comparison.csv        one row per model  -> report §3.1 table
    cv_folds.csv                one row per model x fold
    confusion_matrices.json     student-test confusion matrices
    histgb_loss_history.csv     per-iteration train/validation loss of HistGradientBoosting
    probs/oof_train_<model>.parquet, probs/student_test_<model>.parquet   (for P3's HMM)
Figures (reports/figures/): model_comparison_panel.png, model_overfit_gap.png,
    model_confusion_matrices.png, model_histgb_loss_curve.png
Log: experiments/logs/compare_models_<timestamp>.log
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from pathsnap import io
from pathsnap.evaluation.candidate_metrics import top1_predictions
from pathsnap.evaluation.route_metrics import route_continuity
from pathsnap.models.registry import load_model_configs
from pathsnap.models.train import cross_validate, feature_columns, fit, load_split, score, trusted_only
from pathsnap.viz import plots

RES, FIG, LOGS = ROOT / "experiments" / "results", ROOT / "reports" / "figures", ROOT / "experiments" / "logs"
PROBS = RES / "probs"
for d in (RES, FIG, LOGS, PROBS):
    d.mkdir(parents=True, exist_ok=True)

log_path = LOGS / f"compare_models_{datetime.now():%Y%m%d_%H%M%S}.log"
_log_f = open(log_path, "w")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    _log_f.write(msg + "\n"); _log_f.flush()


def key_config(cfg):
    p = {k: v for k, v in (cfg.get("params") or {}).items() if k not in ("n_jobs", "verbose")}
    s = ", ".join(f"{k}={v}" for k, v in p.items())
    if cfg.get("scale"):
        s += ", scaled"
    if cfg.get("max_train_obs"):
        s += f", trained on {cfg['max_train_obs']} obs subsample"
    return s


def route_continuity_per_trip(test_c, p, raw_dir):
    """Classifier-only route continuity: top-1 arc per observation, in time order, per trip."""
    pred = top1_predictions(test_c, p).sort_values(["trip_id", "obs_idx"])
    vals = {}
    for tid, g in pred.groupby("trip_id"):
        _, arcs = io.load_network(raw_dir / tid)
        net = SimpleNamespace(from_node=arcs["from_node"].to_numpy(), to_node=arcs["to_node"].to_numpy())
        vals[tid] = route_continuity(g["arc_id"].to_numpy(), net)
    return float(np.mean(list(vals.values()))), vals


ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", help="subset of model keys (default: all in configs/models.yaml)")
args = ap.parse_args()

all_cfg = load_model_configs(ROOT / "configs" / "models.yaml")
seed, n_folds = all_cfg.get("seed", 42), all_cfg.get("cv_folds", 5)
names = args.models or list(all_cfg["models"])
label_cfg = yaml.safe_load(open(ROOT / "configs" / "candidates.yaml"))

train_c, train_o = trusted_only(*load_split(ROOT / "data" / "processed", "train"),
                                label_cfg["label_ok_max_dist_m"])
test_c, test_o = load_split(ROOT / "data" / "processed", "student_test")
feats = feature_columns(train_c)
log(f"[data] train (trusted): {train_c.trip_id.nunique()} trips, {len(train_o)} obs, {len(train_c)} rows | "
    f"student_test: {test_c.trip_id.nunique()} trips, {len(test_o)} obs, {len(test_c)} rows")
log(f"[data] features ({len(feats)}): {feats}")
log(f"[setup] seed={seed} folds={n_folds} models={names}")

rows, fold_frames, cms = [], [], {}
for name in names:
    cfg = all_cfg["models"][name]
    log(f"\n=== {name} ({cfg['family']}) | {key_config(cfg)}")
    t0 = time.perf_counter()
    folds, oof = cross_validate(name, cfg, train_c, train_o, feats, n_splits=n_folds, seed=seed, log=log)
    cv_s = time.perf_counter() - t0
    fold_frames.append(folds)

    model, fit_s, n_rows = fit(cfg, train_c, feats, seed=seed)
    test_m, p_test = score(model, test_c, test_o, feats)
    cont_mean, cont_trips = route_continuity_per_trip(test_c, p_test, ROOT / "data" / "raw" / "student_test")
    cms[name] = test_m["confusion_matrix"]

    train_c[["trip_id", "obs_idx", "arc_id", "label"]].assign(p=oof.astype("float32")).to_parquet(
        PROBS / f"oof_train_{name}.parquet", index=False)
    test_c[["trip_id", "obs_idx", "arc_id", "label"]].assign(p=p_test.astype("float32")).to_parquet(
        PROBS / f"student_test_{name}.parquet", index=False)

    if cfg["estimator"] == "hist_gradient_boosting":
        clf = model.named_steps["clf"]
        tr, va = np.abs(clf.train_score_), np.abs(clf.validation_score_)   # sklearn stores -loss
        pd.DataFrame({"iteration": np.arange(len(tr)), "train_log_loss": tr,
                      "validation_log_loss": va}).to_csv(RES / "histgb_loss_history.csv", index=False)
        log(f"  [{name}] boosting stopped after {clf.n_iter_} iterations (early stopping)")

    row = {
        "model": name, "family": cfg["family"], "key_config": key_config(cfg),
        "cv_macro_f1_mean": folds.val_macro_f1.mean(), "cv_macro_f1_std": folds.val_macro_f1.std(),
        "cv_top1_mean": folds.val_top1.mean(), "cv_top1_std": folds.val_top1.std(),
        "cv_log_loss_mean": folds.val_log_loss.mean(),
        "cv_train_top1_mean": folds.train_top1.mean(), "cv_train_macro_f1_mean": folds.train_macro_f1.mean(),
        "test_top1": test_m["top1_accuracy"], "test_top1_covered": test_m["top1_accuracy_covered"],
        "test_macro_f1": test_m["macro_f1"], "test_precision": test_m["precision"],
        "test_recall": test_m["recall"], "test_f1": test_m["f1"], "test_accuracy": test_m["accuracy"],
        "test_log_loss": test_m["log_loss"], "test_route_continuity": cont_mean,
        "inference_ms_per_1k_obs": test_m["inference_ms_per_1k_obs"],
        "cv_fit_s_per_fold": folds.fit_s.mean(), "full_fit_s": fit_s, "cv_total_s": cv_s,
        "train_rows_used": n_rows,
    }
    rows.append(row)
    log(f"  [{name}] CV macroF1={row['cv_macro_f1_mean']:.4f}±{row['cv_macro_f1_std']:.4f} "
        f"| TEST top1={row['test_top1']:.4f} macroF1={row['test_macro_f1']:.4f} "
        f"logloss={row['test_log_loss']:.4f} continuity={cont_mean:.4f} "
        f"| infer={row['inference_ms_per_1k_obs']:.1f}ms/1k obs")

comp = pd.DataFrame(rows)
if args.models and (RES / "model_comparison.csv").exists():      # partial run: update rows in place
    old = pd.read_csv(RES / "model_comparison.csv")
    comp = pd.concat([old[~old.model.isin(names)], comp], ignore_index=True)
    old_f = pd.read_csv(RES / "cv_folds.csv")
    fold_frames = [old_f[~old_f.model.isin(names)]] + fold_frames
    cms = {**json.load(open(RES / "confusion_matrices.json")), **cms}
order = [m for m in all_cfg["models"] if m in set(comp.model)]
comp = comp.set_index("model").loc[order].reset_index()
comp.to_csv(RES / "model_comparison.csv", index=False)
pd.concat(fold_frames, ignore_index=True).to_csv(RES / "cv_folds.csv", index=False)
json.dump({m: cms[m] for m in order}, open(RES / "confusion_matrices.json", "w"), indent=1)

figs = plots.make_all(RES, FIG)
log("\n=== SUMMARY (sorted by CV macro-F1)")
cols = ["model", "cv_macro_f1_mean", "cv_macro_f1_std", "test_top1", "test_macro_f1", "test_log_loss",
        "test_route_continuity", "cv_train_top1_mean", "cv_top1_mean", "inference_ms_per_1k_obs"]
log(comp.sort_values("cv_macro_f1_mean", ascending=False)[cols].round(4).to_string(index=False))
log("\nfigures:", ", ".join(str(f.relative_to(ROOT)) for f in figs))
log("log:", log_path.relative_to(ROOT))
