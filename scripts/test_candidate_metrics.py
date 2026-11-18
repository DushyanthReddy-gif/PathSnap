"""Sanity tests for evaluation/candidate_metrics.py on a synthetic candidate table.
Usage (from repo root): python scripts/test_candidate_metrics.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from pathsnap.evaluation.candidate_metrics import evaluate, top1_mask

rng = np.random.default_rng(42)
K, rows, obs_rows = 6, [], []
for trip in [f"{i:08d}" for i in range(6)]:
    for obs in range(60):
        true_k = rng.integers(K)
        covered = rng.random() > 0.10                    # ~10 % of observations: true arc missing
        arcs = rng.choice(1000, K, replace=False)
        true_arc = arcs[true_k] if covered else 5000 + obs
        for k in range(K):
            rows.append((trip, obs, int(arcs[k]), float(rng.random() * 50), int(covered and k == true_k)))
        obs_rows.append((trip, obs, int(true_arc), covered))
cands = pd.DataFrame(rows, columns=["trip_id", "obs_idx", "arc_id", "dist_m", "label"])
obs_labels = pd.DataFrame(obs_rows, columns=["trip_id", "obs_idx", "true_arc_id", "true_in_candidates"])
cover = obs_labels["true_in_candidates"].mean()
print(f"synthetic: {len(cands)} candidates, {len(obs_labels)} observations, coverage={cover:.3f}")

# 1. perfect model: top-1 = coverage, covered-only top-1 = 1
p_perfect = cands["label"].to_numpy() * 0.98 + 0.01
m, _ = evaluate(cands, p_perfect, obs_labels)
assert abs(m["top1_accuracy"] - cover) < 1e-12 and m["top1_accuracy_covered"] == 1.0
print(f"[perfect] top1={m['top1_accuracy']:.3f} covered={m['top1_accuracy_covered']:.3f} "
      f"macroF1={m['macro_f1']:.3f} logloss={m['log_loss']:.4f}")

# 2. random model: top-1 ~ coverage / K
p_rand = rng.random(len(cands))
m, _ = evaluate(cands, p_rand, obs_labels)
print(f"[random ] top1={m['top1_accuracy']:.3f} (expect ~{cover / K:.3f}) macroF1={m['macro_f1']:.3f} "
      f"logloss={m['log_loss']:.3f}")
assert abs(m["top1_accuracy"] - cover / K) < 0.06

# 3. exactly one top-1 per observation, and it matches a brute-force groupby/idxmax
mask = top1_mask(cands, p_rand)
assert mask.sum() == len(obs_labels)
brute = cands.assign(p=p_rand).groupby(["trip_id", "obs_idx"])["p"].idxmax().to_numpy()
assert set(np.where(mask)[0]) == set(brute)

# 4. confusion matrix totals add up; threshold rule also runs
m_thr, _ = evaluate(cands, p_rand, obs_labels, rule="threshold")
assert sum(map(sum, m["confusion_matrix"])) == len(cands)
print("confusion [[TN, FP], [FN, TP]] (top1 rule):", m["confusion_matrix"])
print("ALL OK")
