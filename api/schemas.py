"""Pydantic schemas for the PathSnap FastAPI service."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status", example="ok")
    model: str = Field(..., description="Loaded model identifier", example="final_model")


class TripListResponse(BaseModel):
    trips: List[str] = Field(..., description="List of available trip IDs")


class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="8-character zero-padded trip ID", example="00000052")


class TimingMetrics(BaseModel):
    load_s: float = Field(..., description="Time taken to load raw files in seconds")
    candidates_s: float = Field(..., description="Time taken for candidate generation in seconds")
    model_s: float = Field(..., description="Time taken for classifier probability inference in seconds")
    viterbi_s: float = Field(..., description="Time taken for HMM Viterbi decoding in seconds")
    total_s: float = Field(..., description="Total pipeline latency in seconds")


class EvaluationMetrics(BaseModel):
    metrics_available: bool = Field(..., description="True if ground truth .route was available for scoring")
    top1_accuracy_hmm: Optional[float] = Field(None, description="Top-1 accuracy using HMM decoding")
    top1_accuracy_classifier_only: Optional[float] = Field(None, description="Top-1 accuracy using classifier argmax only")
    route_continuity: Optional[float] = Field(None, description="Fraction of consecutive route arcs that are connected in network")
    route_precision: Optional[float] = Field(None, description="Precision of reconstructed route against reference route")
    route_recall: Optional[float] = Field(None, description="Recall of reconstructed route against reference route")
    route_f1: Optional[float] = Field(None, description="F1 score of reconstructed route against reference route")
    candidate_miss_rate: Optional[float] = Field(None, description="Fraction of observations where true arc is missing from candidate set")


class PredictResponse(BaseModel):
    trip_id: str = Field(..., description="Trip identifier")
    n_observations: int = Field(..., description="Number of GPS observations")
    gps: List[List[float]] = Field(..., description="List of [lat, lon] coordinates downsampled for map display")
    predicted_route: List[List[List[float]]] = Field(
        ...,
        description="Reconstructed route segments as ordered list of [[lat1, lon1], [lat2, lon2]]",
    )
    network_context: List[List[List[float]]] = Field(
        ...,
        description="Road network arc segments near the trajectory as [[lat1, lon1], [lat2, lon2]]",
    )
    metrics: EvaluationMetrics = Field(..., description="Evaluation metrics computed server-side")
    timings: TimingMetrics = Field(..., description="Latency breakdown in seconds")
