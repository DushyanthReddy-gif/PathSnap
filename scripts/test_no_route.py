"""D6 check: candidates + features must work on a trip folder WITHOUT a .route file
(the API runs them on hidden evaluator trips). Run from repo root:
    set PYTHONPATH=src
    python scripts\\test_no_route.py [trip_id] [split]
"""
import shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import io, features
from pathsnap.geo import LocalProjection
from pathsnap.network import RoadNetwork
from pathsnap.candidates import generate_candidates
from pathsnap.features import build_features

trip = sys.argv[1] if len(sys.argv) > 1 else "00000052"
split = sys.argv[2] if len(sys.argv) > 2 else "student_test"
src = ROOT / "data" / "raw" / split / trip

with tempfile.TemporaryDirectory() as tmp:
    dst = Path(tmp) / trip
    shutil.copytree(src, dst)
    for r in dst.glob("*.route"):
        r.unlink()
    assert not list(dst.glob("*.route")), "route file still present"
    print("copied", trip, "to temp folder without .route:", sorted(p.name for p in dst.iterdir()))

    trk = io.load_track(dst)
    nodes, arcs = io.load_network(dst)
    net = RoadNetwork(nodes, arcs, LocalProjection.from_track(trk))
    cands = generate_candidates(trk, net)
    X = build_features(cands, trk, net)

    print("observations:", len(trk), "| candidate rows:", len(cands), "| feature rows:", len(X))
    cols = getattr(features, "FEATURE_COLUMNS", None)
    if cols is not None:
        missing = [c for c in cols if c not in X.columns]
        assert not missing, f"missing feature columns: {missing}"
        assert not X[cols].isna().any().any(), "NaN in feature columns"
        print("all", len(cols), "FEATURE_COLUMNS present, no NaN")
    else:
        print("FEATURE_COLUMNS not in features.py; columns:", list(X.columns))
    assert len(X) > 0
    print("NO-ROUTE TEST OK")
