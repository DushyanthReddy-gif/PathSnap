"""Bayesian hyperparameter optimisation of the selected model with Optuna (report §4).

Run from anywhere:
    python scripts/tune_model.py

1. Optuna TPE search: every trial = 5-fold GroupKFold CV on the 80 train trips (trusted labels,
   same folds/seed as compare_models.py). Objective = mean CV macro-F1. No student-test or
   evaluator data is used inside the search.
2. Refit the best configuration on all 80 train trips -> artifacts/final_model.joblib
   (via the same code as scripts/export_model.py), score student_test.
3. Tuned vs untuned comparison on identical folds and partitions.
4. Out-of-fold + student-test probabilities of the tuned model for P3's HMM.

Outputs:
    experiments/results/optuna_trials.csv          every trial: params, CV macro-F1/Top-1/log loss, time
    experiments/results/optuna_best_params.json    best hyperparameters (+ fixed ones)
    experiments/results/tuned_vs_untuned.csv       the §4 comparison table
    experiments/results/probs/oof_train_hist_gb_tuned.parquet, student_test_hist_gb_tuned.parquet
    reports/figures/optuna_history.png, optuna_param_importance.png
    experiments/logs/tune_model_<timestamp>.log
"""
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml

from pathsnap.models.registry import load_model_configs
from pathsnap.models.train import cross_validate, feature_columns, fit, load_split, score, trusted_only

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

RES, FIG, LOGS = ROOT / "experiments" / "results", ROOT / "reports" / "figures", ROOT / "experiments" / "logs"
PROBS = RES / "probs"
log_path = LOGS / f"tune_model_{datetime.now():%Y%m%d_%H%M%S}.log"
_log_f = open(log_path, "w")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    _log_f.write(msg + "\n"); _log_f.flush()


tcfg = yaml.safe_load(open(ROOT / "configs" / "optuna.yaml"))
mcfg = load_model_configs(ROOT / "configs" / "models.yaml")["models"][tcfg["model"]]
seed, n_folds = tcfg["seed"], tcfg["cv_folds"]
label_cfg = yaml.safe_load(open(ROOT / "configs" / "candidates.yaml"))
train_c, train_o = trusted_only(*load_split(ROOT / "data" / "processed", "train"), label_cfg["label_ok_max_dist_m"])
test_c, test_o = load_split(ROOT / "data" / "processed", "student_test")
feats = feature_columns(train_c)
log(f"[setup] model={tcfg['model']} sampler={tcfg['sampler']} n_trials={tcfg['n_trials']} folds={n_folds} "
    f"objective=mean CV {tcfg['metric']} | train {train_c.trip_id.nunique()} trips, {len(train_c)} rows")
log(f"[setup] search space: {tcfg['search_space']} | fixed: {tcfg['fixed']}")


def suggest(trial):
    p = dict(tcfg["fixed"])
    for name, s in tcfg["search_space"].items():
        if s["type"] == "float":
            p[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
        else:
            p[name] = trial.suggest_int(name, s["low"], s["high"], log=s.get("log", False))
    return p


def objective(trial):
    params = suggest(trial)
    t0 = time.perf_counter()
    folds, _ = cross_validate(tcfg["model"], mcfg, train_c, train_o, feats, n_splits=n_folds, seed=seed,
                              params_override=params, log=lambda *_: None)
    trial.set_user_attr("cv_macro_f1_std", float(folds.val_macro_f1.std()))
    trial.set_user_attr("cv_top1", float(folds.val_top1.mean()))
    trial.set_user_attr("cv_log_loss", float(folds.val_log_loss.mean()))
    trial.set_user_attr("cv_train_top1", float(folds.train_top1.mean()))
    trial.set_user_attr("seconds", time.perf_counter() - t0)
    value = float(folds.val_macro_f1.mean())
    log(f"  trial {trial.number:2d}: CV macroF1={value:.4f} top1={folds.val_top1.mean():.4f} "
        f"logloss={folds.val_log_loss.mean():.4f} ({time.perf_counter() - t0:.1f}s) "
        + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in trial.params.items()))
    return value


# ---------------------------------------------------------------- 1. search
t_search = time.perf_counter()
study = optuna.create_study(direction=tcfg["direction"], sampler=optuna.samplers.TPESampler(seed=seed),
                            study_name=f"pathsnap_{tcfg['model']}")
study.optimize(objective, n_trials=tcfg["n_trials"])
search_s = time.perf_counter() - t_search

trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
trials.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in trials.columns]
trials = trials.rename(columns={"value": "cv_macro_f1"})
trials.to_csv(RES / "optuna_trials.csv", index=False)
best = {**tcfg["fixed"], **study.best_params}
json.dump(best, open(RES / "optuna_best_params.json", "w"), indent=2)
log(f"\n[search] {len(study.trials)} trials in {search_s:.0f}s | best trial #{study.best_trial.number}: "
    f"CV macroF1={study.best_value:.4f} | params={best}")

# ---------------------------------------------------------------- 2. export final model
subprocess.run([sys.executable, str(ROOT / "scripts" / "export_model.py"), "--model", tcfg["model"],
                "--as", "final", "--params", str(RES / "optuna_best_params.json")], check=True)

# ---------------------------------------------------------------- 3. tuned vs untuned (same folds, same partitions)
rows = {}
for label, override in [("untuned", None), ("tuned", best)]:
    folds, oof = cross_validate(tcfg["model"], mcfg, train_c, train_o, feats, n_splits=n_folds, seed=seed,
                                params_override=override, log=lambda *_: None)
    model, fit_s, _ = fit(mcfg, train_c, feats, seed=seed, params_override=override)
    m, p_test = score(model, test_c, test_o, feats)
    rows[label] = {"cv_macro_f1": folds.val_macro_f1.mean(), "cv_macro_f1_std": folds.val_macro_f1.std(),
                   "cv_top1": folds.val_top1.mean(), "student_test_top1": m["top1_accuracy"],
                   "student_test_macro_f1": m["macro_f1"], "student_test_log_loss": m["log_loss"],
                   "inference_ms_per_1k_obs": m["inference_ms_per_1k_obs"], "fit_s": fit_s,
                   "n_trees": int(model.named_steps["clf"].n_iter_)}
    if label == "tuned":   # probabilities of the deployed model for the HMM (P3)
        train_c[["trip_id", "obs_idx", "arc_id", "label"]].assign(p=oof.astype("float32")).to_parquet(
            PROBS / f"oof_train_{tcfg['model']}_tuned.parquet", index=False)
        test_c[["trip_id", "obs_idx", "arc_id", "label"]].assign(p=p_test.astype("float32")).to_parquet(
            PROBS / f"student_test_{tcfg['model']}_tuned.parquet", index=False)
cmp = pd.DataFrame(rows).T
cmp["hpo_trials"], cmp["hpo_search_seconds"] = len(study.trials), round(search_s, 1)
cmp.to_csv(RES / "tuned_vs_untuned.csv", index_label="version")
log("\n=== TUNED vs UNTUNED (same GroupKFold folds, same student-test partition)")
log(cmp.drop(columns=["hpo_trials", "hpo_search_seconds"]).round(4).T.to_string())

# ---------------------------------------------------------------- 4. figures
fig, ax = plt.subplots(figsize=(6, 3.4))
vals = trials["cv_macro_f1"].to_numpy()
ax.plot(trials["number"], vals, "o", ms=4, alpha=0.6, label="trial")
ax.plot(trials["number"], np.maximum.accumulate(vals), "-", lw=2, label="best so far")
ax.axhline(rows["untuned"]["cv_macro_f1"], ls="--", c="grey", lw=1, label="untuned model (same folds)")
ax.set_xlabel("trial"); ax.set_ylabel("CV macro-F1"); ax.set_title("Optuna TPE optimisation history", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(FIG / "optuna_history.png", dpi=200); plt.close(fig)

imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=seed))
fig, ax = plt.subplots(figsize=(5.5, 3))
ax.barh(list(imp)[::-1], list(imp.values())[::-1], color="#4C72B0")
ax.set_xlabel("importance (fANOVA)"); ax.set_title("Hyperparameter importance", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "optuna_param_importance.png", dpi=200); plt.close(fig)
pd.Series(imp, name="importance").to_csv(RES / "optuna_param_importance.csv", index_label="param")
log(f"\n[importance] {imp}")
log("log:", log_path.relative_to(ROOT))
