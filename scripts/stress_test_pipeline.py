"""Robustness check of the full prediction pipeline on the hardest TRAIN trips.

Run from anywhere:
    python scripts/stress_test_pipeline.py

Why: the system must work on all 10 hidden evaluator trips, which may have very large road
networks, very long trips or long GPS gaps. Those trips must stay untouched before the
evaluation, so we check the worst cases we are allowed to use instead: our own train trips
with the largest network, the most points and the longest gaps.

Checks: no crash, decoder used, time per trip. Writes experiments/results/stress_test.csv.
"""
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from pathsnap import pipeline

CASES = {
    "00000013": "largest network (307,392 arcs)",
    "00000054": "most GPS points (19,313); reference route does not fit the GPS",
    "00000021": "15,783 points, 2,160 s gap",
    "00000090": "longest gap (153,064 s)",
    "00000078": "52,800 s gap, 75% of steps 1 s",
}

art = pipeline.load_artifacts()
rows = []
for trip, why in CASES.items():
    d = ROOT / "data" / "raw" / "train" / trip
    t0 = time.perf_counter()
    try:
        res = pipeline.predict_trip(d, art)
        m = pipeline.score_trip(res, d)
        rows.append({"trip_id": trip, "stress": why, "status": "ok", "seconds": round(time.perf_counter() - t0, 2),
                     "n_obs": res["n_observations"], "route_arcs": len(res["route"]),
                     "top1_hmm": m.get("top1_accuracy_hmm"), "route_continuity": m.get("route_continuity")})
    except Exception as e:
        traceback.print_exc()
        rows.append({"trip_id": trip, "stress": why, "status": f"FAILED: {e!r}",
                     "seconds": round(time.perf_counter() - t0, 2)})

df = pd.DataFrame(rows)
out = ROOT / "experiments" / "results" / "stress_test.csv"
df.to_csv(out, index=False)
print("\n" + df.to_string(index=False))
print(f"\n{'ALL STRESS TESTS PASSED' if (df.status == 'ok').all() else 'SOME STRESS TESTS FAILED'} -> {out}")
