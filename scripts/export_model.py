"""Train one model on all trusted training trips, score it on student_test, save the artifacts.

Run from the repo root:
    python scripts/export_model.py --model hist_gb --as baseline
    python scripts/export_model.py --model hist_gb --as final --params experiments/results/optuna_best_params.json

Writes:
    artifacts/{baseline|final}_model.joblib   fitted sklearn Pipeline (imputer [+ scaler] + model)
    artifacts/model_meta.json                 what pipeline.py needs: model_name, feature_columns, ...
    experiments/results/export_{as}_{model}.json   the same metadata, kept as a result record
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import yaml

from pathsnap.models.registry import load_model_configs
from pathsnap.models.train import feature_columns, fit, load_split, score, trusted_only

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="model key in configs/models.yaml")
ap.add_argument("--as", dest="kind", choices=["baseline", "final"], required=True)
ap.add_argument("--params", help="JSON file with hyperparameters overriding the config (tuned model)")
args = ap.parse_args()

all_cfg = load_model_configs(ROOT / "configs" / "models.yaml")
cfg, seed = all_cfg["models"][args.model], all_cfg.get("seed", 42)
params_path = Path(args.params) if args.params else None
if params_path is not None and not params_path.is_absolute():
    params_path = ROOT / params_path                         # relative paths are from the repo root
override = json.load(open(params_path)) if params_path else None
label_cfg = yaml.safe_load(open(ROOT / "configs" / "candidates.yaml"))
processed = ROOT / "data" / "processed"

train_c, train_o = trusted_only(*load_split(processed, "train"), label_cfg["label_ok_max_dist_m"])
test_c, test_o = load_split(processed, "student_test")           # all observations, no filter
feats = feature_columns(train_c)

model, fit_s, n_rows = fit(cfg, train_c, feats, seed=seed, params_override=override)
test_m, _ = score(model, test_c, test_o, feats)

out_dir = ROOT / "artifacts"
out_dir.mkdir(exist_ok=True)
model_path = out_dir / f"{args.kind}_model.joblib"
joblib.dump(model, model_path, compress=3)

meta = {
    "model_name": args.model,
    "kind": args.kind,
    "family": cfg["family"],
    "estimator": cfg["estimator"],
    "params": {**(cfg.get("params") or {}), **(override or {})},
    "feature_columns": feats,
    "seed": seed,
    "trained_on": f"train ({train_c['trip_id'].nunique()} trips, trusted labels only, {n_rows} candidate rows)",
    "fit_seconds": round(fit_s, 2),
    "model_file_mb": round(model_path.stat().st_size / 1e6, 2),
    "student_test": {k: v for k, v in test_m.items() if k != "confusion_matrix"},
    "student_test_confusion_matrix": test_m["confusion_matrix"],
    "created": datetime.now().isoformat(timespec="seconds"),
}
json.dump(meta, open(out_dir / "model_meta.json", "w"), indent=2)
res = ROOT / "experiments" / "results"
res.mkdir(parents=True, exist_ok=True)
json.dump(meta, open(res / f"export_{args.kind}_{args.model}.json", "w"), indent=2)

print(f"saved {model_path.relative_to(ROOT)} ({meta['model_file_mb']} MB), fit {fit_s:.1f}s on {n_rows} rows")
print(f"student_test: top1={test_m['top1_accuracy']:.4f} macroF1={test_m['macro_f1']:.4f} "
      f"logloss={test_m['log_loss']:.4f} infer={test_m['inference_ms_per_1k_obs']:.1f}ms/1k obs")
