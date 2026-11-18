"""End-to-end smoke test script for the PathSnap pipeline.

Validates:
1. All 10 student_test trips complete without exceptions.
2. Asserts exactly one prediction per GPS observation.
3. Scores each trip against ground truth .route and prints summary metrics.
4. Edge-case test: verifies graceful execution without .route (metrics_available: False).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add src to python path if not installed as package
repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pathsnap.pipeline import load_artifacts, predict_trip, score_trip


def run_smoke_tests() -> bool:
    print("=" * 80)
    print("PathSnap Pipeline Smoke Test Suite")
    print("=" * 80)

    # 1. Load Artifacts
    artifacts_dir = repo_root / "artifacts"
    artifacts = load_artifacts(artifacts_dir)
    print(f"Loaded model identifier: {artifacts.get('model_name')}\n")

    # 2. Locate Student Test Trips
    test_dir = repo_root / "data" / "raw" / "student_test"
    if not test_dir.exists():
        print(f"Error: test directory '{test_dir}' does not exist!")
        return False

    trip_ids = sorted([d.name for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"Found {len(trip_ids)} test trips in {test_dir}: {trip_ids}\n")

    if not trip_ids:
        print("Error: No test trips found!")
        return False

    all_passed = True
    summary_results = []

    print("-" * 80)
    print(f"{'Trip ID':<10} | {'Obs':<6} | {'Top1 HMM':<9} | {'Top1 Cls':<9} | {'Continuity':<10} | {'F1':<6} | {'Time (s)':<8}")
    print("-" * 80)

    for trip_id in trip_ids:
        trip_path = test_dir / trip_id
        try:
            t_start = time.perf_counter()
            pred = predict_trip(trip_path, artifacts)
            t_pred = time.perf_counter() - t_start

            n_obs = pred["n_observations"]
            hmm_df = pred["hmm"]

            # ASSERTION 1: Exactly one prediction per observation
            assert len(hmm_df) == n_obs, f"Trip {trip_id}: expected {n_obs} predictions, got {len(hmm_df)}"

            # ASSERTION 2: Prediction columns contain obs_idx and arc_id
            assert "obs_idx" in hmm_df.columns and "arc_id" in hmm_df.columns

            # Score against ground truth
            metrics = score_trip(pred, trip_path)
            assert metrics.get("metrics_available") is True, f"Trip {trip_id}: metrics should be available"

            top1_hmm = metrics.get("top1_accuracy_hmm", 0.0)
            top1_cls = metrics.get("top1_accuracy_classifier_only", 0.0)
            continuity = metrics.get("route_continuity", 0.0)
            f1 = metrics.get("route_f1", 0.0)

            print(f"{trip_id:<10} | {n_obs:<6} | {top1_hmm:<9.4f} | {top1_cls:<9.4f} | {continuity:<10.4f} | {f1:<6.4f} | {t_pred:<8.2f}")
            summary_results.append({
                "trip_id": trip_id,
                "n_obs": n_obs,
                "top1_hmm": top1_hmm,
                "top1_cls": top1_cls,
                "continuity": continuity,
                "f1": f1,
                "time_s": t_pred,
            })
        except Exception as e:
            print(f"{trip_id:<10} | FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("-" * 80)

    # 3. Edge-Case Test: Trip without .route
    print("\nExecuting Edge-Case Test: Trip without .route file...")
    sample_trip_id = trip_ids[0]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_trip_dir = Path(tmp_dir) / sample_trip_id
        tmp_trip_dir.mkdir(parents=True, exist_ok=True)

        src_trip_dir = test_dir / sample_trip_id
        # Copy only .track, .nodes, .arcs — omit .route
        for suffix in [".track", ".nodes", ".arcs"]:
            src_f = src_trip_dir / f"{sample_trip_id}{suffix}"
            if src_f.exists():
                shutil.copy(src_f, tmp_trip_dir / f"{sample_trip_id}{suffix}")

        try:
            pred_noroute = predict_trip(tmp_trip_dir, artifacts)
            assert len(pred_noroute["hmm"]) == pred_noroute["n_observations"]

            score_noroute = score_trip(pred_noroute, tmp_trip_dir)
            assert score_noroute.get("metrics_available") is False, "metrics_available must be False when .route is absent"
            print("Edge-case test PASSED: Pipeline predicted gracefully with metrics_available: False.")
        except Exception as e:
            print(f"Edge-case test FAILED: {e}")
            all_passed = False

    # Summary
    if all_passed:
        avg_top1_hmm = sum(r["top1_hmm"] for r in summary_results) / len(summary_results)
        avg_continuity = sum(r["continuity"] for r in summary_results) / len(summary_results)
        avg_time = sum(r["time_s"] for r in summary_results) / len(summary_results)
        print("\n" + "=" * 80)
        print("ALL SMOKE TESTS PASSED!")
        print(f"Average Top-1 HMM:     {avg_top1_hmm * 100:.2f}%")
        print(f"Average Continuity:    {avg_continuity * 100:.2f}%")
        print(f"Average Trip Latency:  {avg_time:.2f} s")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("SOME SMOKE TESTS FAILED!")
        print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
