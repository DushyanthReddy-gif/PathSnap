"""Unified map-matching and trajectory reconstruction pipeline.

Provides:
- predict_trip(trip_dir, artifacts) -> dict
- score_trip(prediction, trip_dir) -> dict
- load_artifacts(artifacts_dir) -> dict
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fallback / Built-in Geo & Projection Utilities
# ---------------------------------------------------------------------------

R_EARTH = 6371000.0  # Earth radius in metres


def project_local_meters(
    lons: np.ndarray,
    lats: np.ndarray,
    ref_lon: float,
    ref_lat: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project lon/lat degrees into local Cartesian metres."""
    lat_rad = math.radians(ref_lat)
    cos_lat = math.cos(lat_rad)

    x = np.radians(lons - ref_lon) * R_EARTH * cos_lat
    y = np.radians(lats - ref_lat) * R_EARTH

    return x, y


# ---------------------------------------------------------------------------
# Raw Data Loaders
# ---------------------------------------------------------------------------

def load_raw_trip_data(
    trip_dir: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load .track, .nodes, and .arcs without headers."""

    trip_path = Path(trip_dir)
    trip_id = trip_path.name

    track_file = trip_path / f"{trip_id}.track"
    nodes_file = trip_path / f"{trip_id}.nodes"
    arcs_file = trip_path / f"{trip_id}.arcs"

    if not track_file.exists():
        raise FileNotFoundError(f"Missing track file: {track_file}")

    if not nodes_file.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_file}")

    if not arcs_file.exists():
        raise FileNotFoundError(f"Missing arcs file: {arcs_file}")

    track_df = pd.read_csv(
        track_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "t"],
        dtype={
            "lon": np.float64,
            "lat": np.float64,
            "t": np.float64,
        },
    )

    nodes_df = pd.read_csv(
        nodes_file,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={
            "lon": np.float64,
            "lat": np.float64,
        },
    )

    arcs_df = pd.read_csv(
        arcs_file,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype={
            "from_node": np.int32,
            "to_node": np.int32,
        },
    )

    track_df["obs_idx"] = np.arange(
        len(track_df),
        dtype=np.int32,
    )

    return track_df, nodes_df, arcs_df


def load_route_if_exists(
    trip_dir: str | Path,
) -> Optional[np.ndarray]:
    """Load ground-truth route arc_ids if present.

    MUST ONLY be called during scoring, NEVER in predict_trip.
    """

    trip_path = Path(trip_dir)
    trip_id = trip_path.name
    route_file = trip_path / f"{trip_id}.route"

    if not route_file.exists():
        return None

    try:
        route_arr = (
            pd.read_csv(
                route_file,
                header=None,
            )
            .iloc[:, 0]
            .to_numpy(dtype=np.int32)
        )

        return route_arr

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Candidate Generation Fallback
# ---------------------------------------------------------------------------

def generate_candidates_internal(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    radius_m: float = 50.0,
    max_k: int = 15,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute spatial candidate arcs for each observation."""

    ref_lon = float(track_df["lon"].mean())
    ref_lat = float(track_df["lat"].mean())

    obs_x, obs_y = project_local_meters(
        track_df["lon"].to_numpy(),
        track_df["lat"].to_numpy(),
        ref_lon,
        ref_lat,
    )

    node_x, node_y = project_local_meters(
        nodes_df["lon"].to_numpy(),
        nodes_df["lat"].to_numpy(),
        ref_lon,
        ref_lat,
    )

    from_nodes = arcs_df["from_node"].to_numpy()
    to_nodes = arcs_df["to_node"].to_numpy()

    arc_x1 = node_x[from_nodes]
    arc_y1 = node_y[from_nodes]
    arc_x2 = node_x[to_nodes]
    arc_y2 = node_y[to_nodes]

    dx = arc_x2 - arc_x1
    dy = arc_y2 - arc_y1

    arc_len_sq = dx**2 + dy**2
    arc_len = np.sqrt(arc_len_sq)

    arc_len_sq_safe = np.where(
        arc_len_sq < 1e-6,
        1e-6,
        arc_len_sq,
    )

    arc_mid_x = 0.5 * (arc_x1 + arc_x2)
    arc_mid_y = 0.5 * (arc_y1 + arc_y2)

    n_obs = len(track_df)
    n_arcs = len(arcs_df)

    cand_records = []

    margin = radius_m * 2.0

    min_x = obs_x.min() - margin
    max_x = obs_x.max() + margin
    min_y = obs_y.min() - margin
    max_y = obs_y.max() + margin

    in_bbox_mask = (
        (np.minimum(arc_x1, arc_x2) <= max_x)
        & (np.maximum(arc_x1, arc_x2) >= min_x)
        & (np.minimum(arc_y1, arc_y2) <= max_y)
        & (np.maximum(arc_y1, arc_y2) >= min_y)
    )

    active_arc_indices = np.where(
        in_bbox_mask
    )[0]

    if len(active_arc_indices) == 0:
        active_arc_indices = np.arange(
            min(n_arcs, 500)
        )

    sub_x1 = arc_x1[active_arc_indices]
    sub_y1 = arc_y1[active_arc_indices]
    sub_dx = dx[active_arc_indices]
    sub_dy = dy[active_arc_indices]
    sub_len_sq = arc_len_sq_safe[
        active_arc_indices
    ]
    sub_mid_x = arc_mid_x[
        active_arc_indices
    ]
    sub_mid_y = arc_mid_y[
        active_arc_indices
    ]

    for i in range(n_obs):

        ox = obs_x[i]
        oy = obs_y[i]
        t_val = float(
            track_df["t"].iloc[i]
        )

        coarse_dist_sq = (
            (sub_mid_x - ox) ** 2
            + (sub_mid_y - oy) ** 2
        )

        coarse_mask = (
            coarse_dist_sq
            <= ((radius_m + 150.0) ** 2)
        )

        cand_idx = np.where(
            coarse_mask
        )[0]

        if len(cand_idx) == 0:
            cand_idx = np.argsort(
                coarse_dist_sq
            )[:3]

        c_x1 = sub_x1[cand_idx]
        c_y1 = sub_y1[cand_idx]
        c_dx = sub_dx[cand_idx]
        c_dy = sub_dy[cand_idx]
        c_len_sq = sub_len_sq[cand_idx]

        u = (
            (
                (ox - c_x1) * c_dx
                + (oy - c_y1) * c_dy
            )
            / c_len_sq
        )

        u_clipped = np.clip(
            u,
            0.0,
            1.0,
        )

        px = (
            c_x1
            + u_clipped * c_dx
        )

        py = (
            c_y1
            + u_clipped * c_dy
        )

        dists = np.sqrt(
            (ox - px) ** 2
            + (oy - py) ** 2
        )

        sort_order = np.argsort(dists)
        chosen_order = sort_order[:max_k]

        for rank_idx in chosen_order:

            real_arc_id = int(
                active_arc_indices[
                    cand_idx[rank_idx]
                ]
            )

            dist_val = float(
                dists[rank_idx]
            )

            proj_val = float(
                u_clipped[rank_idx]
            )

            cand_records.append(
                {
                    "obs_idx": i,
                    "t": t_val,
                    "arc_id": real_arc_id,
                    "dist_m": dist_val,
                    "proj_frac": proj_val,
                }
            )

    cands_df = pd.DataFrame(
        cand_records
    )

    context_meta = {
        "ref_lon": ref_lon,
        "ref_lat": ref_lat,
        "active_arcs": active_arc_indices,
    }

    return cands_df, context_meta


# ---------------------------------------------------------------------------
# Viterbi HMM Decoder Fallback
# ---------------------------------------------------------------------------

def viterbi_decode_internal(
    cands_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:

    """Decode optimal sequence of arcs using Viterbi dynamic programming."""

    if params is None:
        params = {}

    sigma_z = params.get(
        "sigma_z",
        10.0,
    )

    penalty_disconnected = params.get(
        "penalty_disconnected",
        -15.0,
    )

    from_nodes = arcs_df[
        "from_node"
    ].to_dict()

    to_nodes = arcs_df[
        "to_node"
    ].to_dict()

    grouped = cands_df.groupby(
        "obs_idx"
    )

    obs_indices = sorted(
        grouped.groups.keys()
    )

    n_steps = len(
        obs_indices
    )

    if n_steps == 0:

        empty_df = pd.DataFrame(
            columns=[
                "obs_idx",
                "arc_id",
            ]
        )

        return (
            empty_df,
            empty_df,
            [],
        )

    classifier_preds = []
    obs_candidates_list = []

    for obs_idx in obs_indices:

        group = grouped.get_group(
            obs_idx
        )

        if "prob" in group.columns:

            best_idx = group[
                "prob"
            ].idxmax()

            best_arc = int(
                group.loc[
                    best_idx,
                    "arc_id",
                ]
            )

        else:

            best_idx = group[
                "dist_m"
            ].idxmin()

            best_arc = int(
                group.loc[
                    best_idx,
                    "arc_id",
                ]
            )

        classifier_preds.append(
            {
                "obs_idx": obs_idx,
                "arc_id": best_arc,
            }
        )

        obs_candidates_list.append(
            group.to_dict(
                "records"
            )
        )

    classifier_only_df = pd.DataFrame(
        classifier_preds
    )

    V: List[
        List[Tuple[float, int]]
    ] = []

    step0_cands = (
        obs_candidates_list[0]
    )

    step0_V = []

    for c in step0_cands:

        arc_id = c[
            "arc_id"
        ]

        dist = c[
            "dist_m"
        ]

        if "prob" in c:

            p_cls = max(
                c["prob"],
                1e-6,
            )

            emis = (
                math.log(p_cls)
                - 0.5
                * (
                    dist
                    / sigma_z
                ) ** 2
            )

        else:

            emis = (
                -0.5
                * (
                    dist
                    / sigma_z
                ) ** 2
            )

        step0_V.append(
            (
                emis,
                -1,
            )
        )

    V.append(
        step0_V
    )

    for t in range(
        1,
        n_steps,
    ):

        prev_cands = (
            obs_candidates_list[
                t - 1
            ]
        )

        curr_cands = (
            obs_candidates_list[
                t
            ]
        )

        curr_V = []

        for c_idx, curr_c in enumerate(
            curr_cands
        ):

            curr_arc = curr_c[
                "arc_id"
            ]

            curr_dist = curr_c[
                "dist_m"
            ]

            if "prob" in curr_c:

                p_cls = max(
                    curr_c[
                        "prob"
                    ],
                    1e-6,
                )

                emis = (
                    math.log(
                        p_cls
                    )
                    - 0.5
                    * (
                        curr_dist
                        / sigma_z
                    ) ** 2
                )

            else:

                emis = (
                    -0.5
                    * (
                        curr_dist
                        / sigma_z
                    ) ** 2
                )

            best_val = -1e9
            best_prev = 0

            for p_idx, prev_c in enumerate(
                prev_cands
            ):

                prev_val = V[
                    t - 1
                ][p_idx][0]

                prev_arc = prev_c[
                    "arc_id"
                ]

                if (
                    prev_arc
                    == curr_arc
                ):

                    trans = 0.0

                elif (
                    to_nodes.get(
                        prev_arc
                    )
                    == from_nodes.get(
                        curr_arc
                    )
                ):

                    trans = -1.0

                elif (
                    from_nodes.get(
                        prev_arc
                    )
                    == to_nodes.get(
                        curr_arc
                    )
                ):

                    trans = -3.0

                else:

                    trans = (
                        penalty_disconnected
                    )

                total_p = (
                    prev_val
                    + trans
                    + emis
                )

                if (
                    total_p
                    > best_val
                ):

                    best_val = total_p
                    best_prev = p_idx

            curr_V.append(
                (
                    best_val,
                    best_prev,
                )
            )

        V.append(
            curr_V
        )

    best_last_idx = int(
        np.argmax(
            [
                v[0]
                for v in V[-1]
            ]
        )
    )

    opt_states = [
        best_last_idx
    ]

    for t in range(
        n_steps - 1,
        0,
        -1,
    ):

        prev_idx = V[t][
            opt_states[-1]
        ][1]

        opt_states.append(
            prev_idx
        )

    opt_states.reverse()

    hmm_preds = []

    for t in range(
        n_steps
    ):

        chosen_cand = (
            obs_candidates_list[
                t
            ][
                opt_states[t]
            ]
        )

        hmm_preds.append(
            {
                "obs_idx": obs_indices[t],
                "arc_id": int(
                    chosen_cand[
                        "arc_id"
                    ]
                ),
            }
        )

    hmm_df = pd.DataFrame(
        hmm_preds
    )

    route_arcs: List[int] = []

    for row in hmm_preds:

        arc = row[
            "arc_id"
        ]

        if (
            not route_arcs
            or route_arcs[-1]
            != arc
        ):

            route_arcs.append(
                arc
            )

    return (
        classifier_only_df,
        hmm_df,
        route_arcs,
    )


# ---------------------------------------------------------------------------
# Artifact Loading
# ---------------------------------------------------------------------------

def load_artifacts(
    artifacts_dir: str | Path = "artifacts",
) -> Dict[str, Any]:
    """Load model, metadata, and HMM parameters once at startup."""
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.is_absolute():        # relative -> repo root (works from any cwd)
        artifacts_path = Path(__file__).resolve().parents[2] / artifacts_path
    artifacts: Dict[str, Any] = {
        "model": None,
        "meta": {},
        "hmm_params": {},
        "model_name": "Baseline-Distance-HMM",
    }

    final_model_path = (
        artifacts_path
        / "final_model.joblib"
    )

    baseline_model_path = (
        artifacts_path
        / "baseline_model.joblib"
    )

    meta_path = (
        artifacts_path
        / "model_meta.json"
    )

    hmm_path = (
        artifacts_path
        / "hmm_params.json"
    )

    import json

    if meta_path.exists():

        try:

            with open(
                meta_path,
                "r",
                encoding="utf-8",
            ) as f:

                artifacts[
                    "meta"
                ] = json.load(f)

            if (
                "model_name"
                in artifacts[
                    "meta"
                ]
            ):

                artifacts[
                    "model_name"
                ] = artifacts[
                    "meta"
                ][
                    "model_name"
                ]

        except Exception as e:

            print(
                "[META LOAD] "
                f"failed={meta_path.name} "
                f"reason={e}"
            )

    if hmm_path.exists():

        try:

            with open(
                hmm_path,
                "r",
                encoding="utf-8",
            ) as f:

                artifacts[
                    "hmm_params"
                ] = json.load(f)

        except Exception as e:

            print(
                "[HMM PARAMS LOAD] "
                f"failed={hmm_path.name} "
                f"reason={e}"
            )

    import joblib

    for m_path in [
        final_model_path,
        baseline_model_path,
    ]:

        if m_path.exists():

            try:

                artifacts[
                    "model"
                ] = joblib.load(
                    m_path
                )

                artifacts[
                    "model_name"
                ] = m_path.stem

                print(
                    "[MODEL LOAD] "
                    f"model={m_path.stem} "
                    "status=success"
                )

                break

            except Exception as e:

                print(
                    "[MODEL LOAD] "
                    f"failed={m_path.name} "
                    f"reason={e}"
                )

    if artifacts[
        "model"
    ] is None:

        print(
            "[MODEL LOAD] "
            "status=FAILED "
            "reason=no_model_loaded"
        )

    return artifacts


# ---------------------------------------------------------------------------
# Central predict_trip Pipeline
# ---------------------------------------------------------------------------

def predict_trip(
    trip_dir: str | Path,
    artifacts: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """Execute end-to-end reconstruction for a single trip.

    MUST NOT read .route.
    """

    total_start = time.perf_counter()

    trip_path = Path(
        trip_dir
    )

    if artifacts is None:
        artifacts = load_artifacts()

    # -----------------------------------------------------------------------
    # Stage 1: Load Data
    # -----------------------------------------------------------------------

    t0 = time.perf_counter()

    try:

        from pathsnap.io import (
            load_track,
            load_network,
        )

        track_df = load_track(
            trip_path
        )

        nodes_df, arcs_df = (
            load_network(
                trip_path
            )
        )

    except Exception as e:

        print(
            "[DATA] "
            f"fallback_loader "
            f"reason={e}"
        )

        (
            track_df,
            nodes_df,
            arcs_df,
        ) = load_raw_trip_data(
            trip_path
        )

    load_s = (
        time.perf_counter()
        - t0
    )

    # -----------------------------------------------------------------------
    # Stage 2: Candidates & Features
    # -----------------------------------------------------------------------

    t0 = time.perf_counter()

    # Use the exact same pipeline components used during training.
    from pathsnap.geo import LocalProjection
    from pathsnap.network import RoadNetwork
    from pathsnap.candidates import (
        generate_candidates,
    )
    from pathsnap.features import (
        build_features,
    )

    # Same projection used for candidate generation
    proj = LocalProjection.from_track(
        track_df
    )

    # Build road network
    net = RoadNetwork(
        nodes_df,
        arcs_df,
        proj,
    )

    # IMPORTANT:
    # This uses the team's real candidate generator.
    # Training configuration is 50 m radius / K=8.
    cands_df = generate_candidates(
        track_df,
        net,
    )

    print(
        "[CANDIDATES] "
        f"generated={len(cands_df)} "
        f"observations="
        f"{len(track_df)}"  # obs_idx is the index of track_df, not a column
    )

    # Build exactly the feature columns
    # expected by the trained model.
    cands_df = build_features(
        cands_df,
        track_df,
        net,
    )

    print(
        "[FEATURES] "
        f"columns={len(cands_df.columns)} "
        "features_ready=True"
    )

    candidates_s = (
        time.perf_counter()
        - t0
    )

    # -----------------------------------------------------------------------
    # Stage 3: Classifier Inference
    # -----------------------------------------------------------------------

    t0 = time.perf_counter()

    model = artifacts.get(
        "model"
    )

    meta = artifacts.get(
        "meta",
        {},
    )

    feat_cols = meta.get(
        "feature_columns",
        [],
    )

    # DO NOT silently fall back.
    # If the trained model cannot run,
    # fail loudly so the API cannot pretend
    # that the trained model was used.

    if model is None:

        raise RuntimeError(
            "Trained model is not loaded. "
            "Cannot run model inference."
        )

    if not feat_cols:

        raise RuntimeError(
            "model_meta.json does not contain "
            "'feature_columns'."
        )

    missing_features = [
        col
        for col in feat_cols
        if col not in cands_df.columns
    ]

    if missing_features:

        raise RuntimeError(
            "Missing trained-model feature "
            f"columns: {missing_features}"
        )

    X = cands_df[
        feat_cols
    ]

    try:

        cands_df[
            "prob"
        ] = model.predict_proba(
            X
        )[:, 1]

        print(
            "[MODEL] "
            f"model={artifacts.get('model_name', 'trained_model')} "
            "predict_proba=success "
            f"rows={len(X)}"
        )

    except Exception as e:

        raise RuntimeError(
            "Trained model "
            "predict_proba failed: "
            f"{e}"
        ) from e

    model_s = (
        time.perf_counter()
        - t0
    )

    # -----------------------------------------------------------------------
    # Stage 4: HMM / Viterbi Decoding
    # -----------------------------------------------------------------------

    t0 = time.perf_counter()

    classifier_only_df: Optional[
        pd.DataFrame
    ] = None

    hmm_df: Optional[
        pd.DataFrame
    ] = None

    route_arcs: Optional[
        List[int]
    ] = None

    # Try team decoder
    try:

        from pathsnap.hmm.viterbi import (
            decode as team_decode,
        )

        # Leka's decoder reads the classifier probability from column "p"
        # and needs the RoadNetwork (arc lengths, connectivity), not the raw arcs table.
        cands_df["p"] = cands_df["prob"]
        hmm_df = team_decode(
            cands_df,
            net,
            artifacts.get(
                "hmm_params",
                {},
            ),
        )

        route_arcs = []

        for arc in hmm_df[
            "arc_id"
        ]:

            if (
                not route_arcs
                or route_arcs[-1]
                != arc
            ):

                route_arcs.append(
                    int(arc)
                )

        print(
            "[HMM] "
            "decoder=team "
            "status=success"
        )

    except Exception as e:

        print(
            "[HMM] "
            "team_decoder_failed "
            f"reason={e}"
        )

    if (
        hmm_df is None
        or route_arcs is None
    ):

        print(
            "[HMM] "
            "decoder=internal_fallback"
        )

        (
            classifier_only_df,
            hmm_df,
            route_arcs,
        ) = viterbi_decode_internal(
            cands_df,
            arcs_df,
            artifacts.get(
                "hmm_params",
                {},
            ),
        )

    elif (
        classifier_only_df
        is None
    ):

        best_rows = cands_df.loc[
            cands_df.groupby(
                "obs_idx"
            )[
                "prob"
            ].idxmax()
        ]

        classifier_only_df = (
            best_rows[
                [
                    "obs_idx",
                    "arc_id",
                ]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    viterbi_s = (
        time.perf_counter()
        - t0
    )

    total_s = (
        time.perf_counter()
        - total_start
    )

    # -----------------------------------------------------------------------
    # Prepare Display Context Geometry
    # -----------------------------------------------------------------------

    gps_coords = (
        track_df[
            [
                "lat",
                "lon",
            ]
        ]
        .to_numpy()
        .tolist()
    )

    node_lats = (
        nodes_df[
            "lat"
        ]
        .to_numpy()
    )

    node_lons = (
        nodes_df[
            "lon"
        ]
        .to_numpy()
    )

    from_nodes = (
        arcs_df[
            "from_node"
        ]
        .to_numpy()
    )

    to_nodes = (
        arcs_df[
            "to_node"
        ]
        .to_numpy()
    )

    predicted_segments = []

    for arc_id in route_arcs:

        if (
            0 <= arc_id
            < len(arcs_df)
        ):

            fn = from_nodes[
                arc_id
            ]

            tn = to_nodes[
                arc_id
            ]

            predicted_segments.append(
                [
                    [
                        float(
                            node_lats[
                                fn
                            ]
                        ),
                        float(
                            node_lons[
                                fn
                            ]
                        ),
                    ],
                    [
                        float(
                            node_lats[
                                tn
                            ]
                        ),
                        float(
                            node_lons[
                                tn
                            ]
                        ),
                    ],
                ]
            )

    cand_arc_ids = (
        cands_df[
            "arc_id"
        ]
        .unique()
    )

    network_context = []

    for arc_id in cand_arc_ids:

        if (
            0 <= arc_id
            < len(arcs_df)
        ):

            fn = from_nodes[
                arc_id
            ]

            tn = to_nodes[
                arc_id
            ]

            network_context.append(
                [
                    [
                        float(
                            node_lats[
                                fn
                            ]
                        ),
                        float(
                            node_lons[
                                fn
                            ]
                        ),
                    ],
                    [
                        float(
                            node_lats[
                                tn
                            ]
                        ),
                        float(
                            node_lons[
                                tn
                            ]
                        ),
                    ],
                ]
            )

    return {
        "classifier_only":
            classifier_only_df,

        "hmm":
            hmm_df,

        "route":
            route_arcs,

        "timings": {
            "load_s":
                round(
                    load_s,
                    3,
                ),

            "candidates_s":
                round(
                    candidates_s,
                    3,
                ),

            "model_s":
                round(
                    model_s,
                    3,
                ),

            "viterbi_s":
                round(
                    viterbi_s,
                    3,
                ),

            "total_s":
                round(
                    total_s,
                    3,
                ),
        },

        "gps":
            gps_coords,

        "predicted_route_segments":
            predicted_segments,

        "network_context":
            network_context,

        "n_observations":
            len(track_df),

        "cands_df":
            cands_df,
    }


# ---------------------------------------------------------------------------
# Scoring against ground-truth .route
# SERVER-SIDE ONLY
# ---------------------------------------------------------------------------

def score_trip(
    prediction: Dict[str, Any],
    trip_dir: str | Path,
) -> Dict[str, Any]:
    """Score prediction against .route ground truth."""

    trip_path = Path(
        trip_dir
    )

    ref_route = load_route_if_exists(
        trip_path
    )

    if (
        ref_route is None
        or len(ref_route) == 0
    ):

        return {
            "metrics_available": False
        }

    try:

        (
            _,
            nodes_df,
            arcs_df,
        ) = load_raw_trip_data(
            trip_path
        )

    except Exception:

        return {
            "metrics_available": False
        }

    pred_route = prediction.get(
        "route",
        [],
    )

    hmm_df = prediction.get(
        "hmm"
    )

    cls_df = prediction.get(
        "classifier_only"
    )

    cands_df = prediction.get(
        "cands_df"
    )

    ref_set = set(
        ref_route
    )

    pred_set = set(
        pred_route
    )

    overlap = len(
        ref_set.intersection(
            pred_set
        )
    )

    precision = (
        float(
            overlap
            / len(pred_set)
        )
        if pred_set
        else 0.0
    )

    recall = (
        float(
            overlap
            / len(ref_set)
        )
        if ref_set
        else 0.0
    )

    f1 = (
        float(
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )

    from_nodes = (
        arcs_df[
            "from_node"
        ]
        .to_numpy()
    )

    to_nodes = (
        arcs_df[
            "to_node"
        ]
        .to_numpy()
    )

    continuity = 1.0

    if len(pred_route) > 1:

        valid_transitions = 0

        total_transitions = (
            len(pred_route)
            - 1
        )

        for i in range(
            total_transitions
        ):

            a1 = pred_route[
                i
            ]

            a2 = pred_route[
                i + 1
            ]

            if (
                0 <= a1
                < len(arcs_df)
                and
                0 <= a2
                < len(arcs_df)
            ):

                if (
                    to_nodes[a1]
                    == from_nodes[a2]
                    or a1 == a2
                ):

                    valid_transitions += 1

        continuity = float(
            valid_transitions
            / total_transitions
        )

    top1_hmm = 0.0
    top1_cls = 0.0
    candidate_miss_rate = 0.0

    if (
        hmm_df is not None
        and not hmm_df.empty
    ):

        n_obs = len(
            hmm_df
        )

        hmm_in_ref = sum(
            1
            for a in hmm_df[
                "arc_id"
            ]
            if a in ref_set
        )

        top1_hmm = float(
            hmm_in_ref
            / n_obs
        )

    if (
        cls_df is not None
        and not cls_df.empty
    ):

        n_obs = len(
            cls_df
        )

        cls_in_ref = sum(
            1
            for a in cls_df[
                "arc_id"
            ]
            if a in ref_set
        )

        top1_cls = float(
            cls_in_ref
            / n_obs
        )

    if (
        cands_df is not None
        and not cands_df.empty
    ):

        obs_groups = (
            cands_df.groupby(
                "obs_idx"
            )[
                "arc_id"
            ]
            .apply(set)
        )

        missed = sum(
            1
            for c_set
            in obs_groups
            if not c_set.intersection(
                ref_set
            )
        )

        candidate_miss_rate = (
            float(
                missed
                / len(
                    obs_groups
                )
            )
            if len(
                obs_groups
            ) > 0
            else 0.0
        )

    return {
        "metrics_available": True,

        "top1_accuracy_hmm":
            round(
                top1_hmm,
                4,
            ),

        "top1_accuracy_classifier_only":
            round(
                top1_cls,
                4,
            ),

        "route_continuity":
            round(
                continuity,
                4,
            ),

        "route_precision":
            round(
                precision,
                4,
            ),

        "route_recall":
            round(
                recall,
                4,
            ),

        "route_f1":
            round(
                f1,
                4,
            ),

        "candidate_miss_rate":
            round(
                candidate_miss_rate,
                4,
            ),
    }