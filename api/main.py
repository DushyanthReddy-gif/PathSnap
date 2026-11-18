"""FastAPI application for PathSnap map matching and route reconstruction service."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

# Ensure src is on sys.path even without pip install -e .
_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import yaml
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.logging_config import setup_api_logger
from api.schemas import (
    EvaluationMetrics,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TimingMetrics,
    TripListResponse,
)
from pathsnap.pipeline import load_artifacts, predict_trip, score_trip

# ---------------------------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("configs/api.yaml")


def load_config() -> Dict[str, Any]:
    default_config = {
        "host": "0.0.0.0",
        "port": 8000,
        "trips_dir": "data/raw/student_test",
        "artifacts_dir": "artifacts",
        "display_max_gps_points": 2000,
        "display_max_network_segments": 3000,
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    default_config.update(loaded)
        except Exception as e:
            print(f"Warning: Failed to parse {CONFIG_PATH}, using defaults: {e}")
    return default_config


config = load_config()
logger = setup_api_logger()

# Global state loaded once at startup
app_state: Dict[str, Any] = {
    "artifacts": None,
    "model_name": "None",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, metadata and HMM parameters once at application startup."""
    artifacts_dir = config.get("artifacts_dir", "artifacts")
    logger.info(f"Loading pipeline artifacts from '{artifacts_dir}'...")
    artifacts = load_artifacts(artifacts_dir)
    app_state["artifacts"] = artifacts
    app_state["model_name"] = artifacts.get("model_name", "Baseline-Distance-HMM")
    logger.info(f"Application initialized successfully with model '{app_state['model_name']}'.")
    yield
    logger.info("Application shutting down.")


# ---------------------------------------------------------------------------
# FastAPI App Instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PathSnap Map-Matching Service",
    description="Live trajectory map-matching and route reconstruction API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware allowing all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return health status and currently loaded model name."""
    return HealthResponse(
        status="ok",
        model=app_state.get("model_name", "unknown"),
    )


@app.get("/trips", response_model=TripListResponse)
def list_trips() -> TripListResponse:
    """Dynamically list all trip IDs found in configured trips_dir."""
    # Re-read config trips_dir in case it was updated on disk
    curr_config = load_config()
    trips_dir = Path(curr_config.get("trips_dir", "data/raw/student_test"))

    if not trips_dir.exists():
        logger.warning(f"Trips directory '{trips_dir}' does not exist.")
        return TripListResponse(trips=[])

    trip_ids: List[str] = []
    for item in sorted(os.listdir(trips_dir)):
        item_path = trips_dir / item
        if item_path.is_dir() and not item.startswith("."):
            trip_ids.append(item)

    return TripListResponse(trips=trip_ids)


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """Execute live end-to-end reconstruction for a given trip_id."""
    curr_config = load_config()
    trips_dir = Path(curr_config.get("trips_dir", "data/raw/student_test"))
    trip_id = request.trip_id.strip()

    trip_path = trips_dir / trip_id
    if not trip_path.exists() or not trip_path.is_dir():
        logger.warning(f"Trip '{trip_id}' not found in '{trips_dir}'")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip ID '{trip_id}' not found in '{trips_dir}'.",
        )

    # Execute prediction pipeline LIVE (never cached)
    artifacts = app_state.get("artifacts")
    try:
        prediction = predict_trip(trip_path, artifacts)
    except Exception as e:
        logger.error(f"Error executing predict_trip for {trip_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction pipeline failed for trip '{trip_id}': {str(e)}",
        )

    # Compute server-side metrics if ground truth .route is available
    metrics_dict = score_trip(prediction, trip_path)

    # Downsample GPS points for UI display if needed
    max_gps = int(curr_config.get("display_max_gps_points", 2000))
    raw_gps = prediction.get("gps", [])
    if len(raw_gps) > max_gps:
        step = max(1, len(raw_gps) // max_gps)
        sampled_gps = raw_gps[::step]
    else:
        sampled_gps = raw_gps

    # Cap network context segments for UI display if needed
    max_network = int(curr_config.get("display_max_network_segments", 3000))
    raw_network = prediction.get("network_context", [])
    if len(raw_network) > max_network:
        sampled_network = raw_network[:max_network]
    else:
        sampled_network = raw_network

    timings = prediction.get("timings", {})
    n_obs = prediction.get("n_observations", 0)

    # Mandatory Request Terminal Logging:
    # [PREDICT] trip=00000052 n_obs=262 cands=1.2s model=0.1s viterbi=0.4s top1_hmm=0.91 continuity=0.98 total=1.9s
    cands_s = timings.get("candidates_s", 0.0)
    model_s = timings.get("model_s", 0.0)
    viterbi_s = timings.get("viterbi_s", 0.0)
    total_s = timings.get("total_s", 0.0)
    top1_hmm = metrics_dict.get("top1_accuracy_hmm", "N/A")
    continuity = metrics_dict.get("route_continuity", "N/A")

    log_line = (
        f"[PREDICT] trip={trip_id} n_obs={n_obs} "
        f"cands={cands_s:.1f}s model={model_s:.1f}s viterbi={viterbi_s:.1f}s "
        f"top1_hmm={top1_hmm} continuity={continuity} total={total_s:.1f}s"
    )
    logger.info(log_line)

    return PredictResponse(
        trip_id=trip_id,
        n_observations=n_obs,
        gps=sampled_gps,
        predicted_route=prediction.get("predicted_route_segments", []),
        network_context=sampled_network,
        metrics=EvaluationMetrics(**metrics_dict),
        timings=TimingMetrics(**timings),
    )
