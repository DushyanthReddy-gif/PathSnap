"""
Viterbi decoder for HMM-based map matching.

Public API
----------
decode(cands, network, params) -> pd.DataFrame[obs_idx, arc_id]

Exactly one row per observation of the input trip.
  - arc_id == -1  if the observation has no candidates.

Design notes
------------
* Fully in log-space.
* GPS gaps (Δt > gap_s or implied speed > max_speed_ms) split the trip into
  independent segments decoded separately.
* If an entire Viterbi step is unreachable (all transitions are LARGE_PENALTY
  AND all subsequent emissions remain -inf), the chain is broken and Viterbi
  restarts from that observation.
* Downsampling: if a trip has > max_obs observations, we decode on a
  time-strided subsample and assign skipped observations from their nearest
  decoded neighbour — the function still returns one row per observation.

Viterbi recurrence (log-space):
    δ_t(j) = max_i [ δ_{t-1}(i) + log A_ij ] + log B_j(o_t)
    ψ_t(j) = argmax_i [ δ_{t-1}(i) + log A_ij ]

Reference: Newson & Krumm (2009), "Hidden Markov Map Matching Through Noise
and Sparseness", ACM GIS.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .emissions import build_log_emission_matrix
from .transitions import (
    LARGE_PENALTY,
    build_transition_matrix,
    clear_cache,
)

# ── default parameter values ──────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "sigma_m": 20.0,       # GPS noise std-dev (metres)
    "beta": 20.0,          # transition smoothness (metres)
    "alpha": 0.0,          # classifier blend weight (0 = distance-only v1)
    "gap_s": 60.0,         # Δt threshold (seconds) to split into segments
    "max_speed_ms": 50.0,  # ~180 km/h; impossible speed triggers a gap
    "cutoff_m": 2000.0,    # Dijkstra cutoff (metres)
    "max_obs": 5000,       # downsample above this for speed
}

_NEG_INF = -1e18  # stand-in for -∞ in arrays


# ── helpers ───────────────────────────────────────────────────────────────

def _gps_dist(lon1, lat1, lon2, lat2) -> float:
    """Approximate Euclidean distance in metres using small-angle (projected)."""
    # Caller passes projected metres if the network has already projected them;
    # otherwise this is a rough value only used to find the cutoff.
    return math.hypot(lon2 - lon1, lat2 - lat1)


def _split_segments(
    obs: pd.DataFrame, gap_s: float, max_speed_ms: float
) -> list[pd.DataFrame]:
    """Split a sorted observation table into continuous segments.

    Splits when:
      * Δt > gap_s, OR
      * implied speed > max_speed_ms (using Euclidean distance in degrees).

    For the speed check we use a rough 111_000 m/deg factor; the transitions
    module uses the actual projected distances.
    """
    if len(obs) == 0:
        return []

    segments: list[pd.DataFrame] = []
    start = 0
    times = obs["t"].to_numpy(dtype=np.float64)
    lons = obs["lon"].to_numpy(dtype=np.float64) if "lon" in obs.columns else np.zeros(len(obs))
    lats = obs["lat"].to_numpy(dtype=np.float64) if "lat" in obs.columns else np.zeros(len(obs))

    for i in range(1, len(obs)):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            dt = 1e-6
        d = math.hypot(lons[i] - lons[i - 1], lats[i] - lats[i - 1]) * 111_000.0
        speed = d / dt

        if dt > gap_s or speed > max_speed_ms:
            segments.append(obs.iloc[start:i].copy())
            start = i

    segments.append(obs.iloc[start:].copy())
    return segments


# ── core Viterbi ──────────────────────────────────────────────────────────

def _viterbi_segment(
    obs_idxs: list[int],
    cand_map: dict[int, list[int]],      # obs_idx → [arc_id, ...]
    log_em: dict[int, dict[int, float]], # obs_idx → {arc_id: log_p}
    times: dict[int, float],             # obs_idx → timestamp
    lons: dict[int, float],
    lats: dict[int, float],
    network,
    beta: float,
    cutoff_m: float,
) -> dict[int, int]:
    """
    Run Viterbi on one continuous segment.

    Returns
    -------
    dict: obs_idx → decoded arc_id  (-1 when no candidates).

    Chain restart: if at any step all delta values are ≤ _NEG_INF (the step
    is unreachable), Viterbi is restarted from that observation.
    """
    result: dict[int, int] = {}

    sub_start = 0
    while sub_start < len(obs_idxs):
        # Find the first obs that has candidates to anchor the chain.
        anchor = sub_start
        while anchor < len(obs_idxs) and not cand_map.get(obs_idxs[anchor]):
            result[obs_idxs[anchor]] = -1
            anchor += 1
        if anchor >= len(obs_idxs):
            break

        sub_obs = obs_idxs[anchor:]
        # --- initialise ---
        first = sub_obs[0]
        states = cand_map[first]  # list[arc_id]
        delta = np.array(
            [log_em[first].get(a, _NEG_INF) for a in states], dtype=np.float64
        )
        # psi[t] = list of back-pointer indices into previous state list
        psi_arcs: list[list[int]] = [states]  # psi_arcs[0] = initial states
        psi_ptr: list[np.ndarray] = []         # psi_ptr[t] = argmax over prev

        chain_obs = [first]
        restarted_at: int | None = None

        for step_i in range(1, len(sub_obs)):
            oi = sub_obs[step_i]
            curr_states = cand_map.get(oi, [])

            if not curr_states:
                # No candidates — emit -1, keep chain so far, decode later
                result[oi] = -1
                chain_obs.append(oi)
                psi_arcs.append([])
                psi_ptr.append(np.array([], dtype=np.int32))
                delta = np.array([], dtype=np.float64)
                # We'll restart Viterbi after this — treat next obs as new start.
                restarted_at = step_i + 1
                break

            # GPS distance for transition
            dt = times[oi] - times[chain_obs[-1]]
            gps_d = math.hypot(
                lons[oi] - lons[chain_obs[-1]],
                lats[oi] - lats[chain_obs[-1]],
            ) * 111_000.0  # rough metres from degrees

            prev_states = psi_arcs[-1]
            if prev_states:
                trans = build_transition_matrix(
                    network, prev_states, curr_states, gps_d, beta, cutoff_m
                )
                # delta_prev  shape (|prev|,), trans shape (|prev|, |curr|)
                scores = delta[:, None] + trans  # broadcast
                best_prev_idx = np.argmax(scores, axis=0)
                new_delta = scores[best_prev_idx, np.arange(len(curr_states))]
            else:
                # Previous step had no candidates (should not happen here)
                best_prev_idx = np.zeros(len(curr_states), dtype=np.int32)
                new_delta = np.full(len(curr_states), _NEG_INF, dtype=np.float64)

            # Add emissions
            em_curr = np.array(
                [log_em[oi].get(a, _NEG_INF) for a in curr_states], dtype=np.float64
            )
            new_delta = new_delta + em_curr

            # Check for complete disconnection → restart
            if np.all(new_delta <= _NEG_INF + 1):
                # Decode what we have so far, then restart
                _backtrack(chain_obs, psi_arcs, psi_ptr, delta, result)
                restarted_at = step_i
                break

            psi_arcs.append(curr_states)
            psi_ptr.append(best_prev_idx.astype(np.int32))
            chain_obs.append(oi)
            delta = new_delta

        else:
            # Normal end of segment — backtrack
            _backtrack(chain_obs, psi_arcs, psi_ptr, delta, result)
            restarted_at = None

        # Advance sub_start
        if restarted_at is None:
            break
        else:
            sub_start = anchor + restarted_at

    return result


def _backtrack(
    chain_obs: list[int],
    psi_arcs: list[list[int]],
    psi_ptr: list[np.ndarray],
    final_delta: np.ndarray,
    result: dict[int, int],
) -> None:
    """Fill result dict by following backpointers from the last step."""
    if len(psi_arcs) == 0 or len(psi_arcs[-1]) == 0:
        for oi in chain_obs:
            result.setdefault(oi, -1)
        return

    T = len(chain_obs)
    # Best state at last step
    if len(final_delta) == 0:
        for oi in chain_obs:
            result.setdefault(oi, -1)
        return

    best_last = int(np.argmax(final_delta))
    path_indices = [best_last]
    cur = best_last

    for t in range(T - 1, 0, -1):
        ptr = psi_ptr[t - 1]  # backpointer array for step t
        if len(ptr) == 0 or cur >= len(ptr):
            # Gap step — stay
            cur = 0
        else:
            cur = int(ptr[cur])
        path_indices.append(cur)

    path_indices.reverse()

    for step_i, oi in enumerate(chain_obs):
        states_at_step = psi_arcs[step_i]
        if not states_at_step:
            result.setdefault(oi, -1)
        else:
            idx = path_indices[step_i] if step_i < len(path_indices) else 0
            idx = max(0, min(idx, len(states_at_step) - 1))
            result.setdefault(oi, states_at_step[idx])


# ── downsampling ──────────────────────────────────────────────────────────

def _downsample_and_assign(
    obs_idxs: list[int],
    decoded: dict[int, int],
    max_obs: int,
) -> dict[int, int]:
    """
    Assign decoded arc_id of nearest decoded neighbour to any obs_idx not
    in `decoded`.  Returns a complete dict for all obs_idxs.
    """
    full: dict[int, int] = {}
    for oi in obs_idxs:
        if oi in decoded:
            full[oi] = decoded[oi]

    # Fill gaps from nearest decoded neighbour (simple forward/backward fill)
    prev_arc = -1
    for oi in obs_idxs:
        if oi in full:
            prev_arc = full[oi]
        else:
            full[oi] = prev_arc

    # Backward fill for any leading -1 from non-decoded start
    next_arc = -1
    for oi in reversed(obs_idxs):
        if full.get(oi, -1) != -1:
            next_arc = full[oi]
        elif next_arc != -1:
            full[oi] = next_arc

    return full


# ── public API ────────────────────────────────────────────────────────────

def decode(
    cands: pd.DataFrame,
    network,
    params: dict | None = None,
) -> pd.DataFrame:
    """
    Viterbi HMM decoder.

    Parameters
    ----------
    cands   : DataFrame with columns  trip_id, obs_idx, t, arc_id, dist_m
              and optionally  p  (classifier probability), lon, lat.
              Contains ALL candidates for a *single* trip.
    network : RoadNetwork for the trip.
    params  : decoder hyper-parameters (dict).  Missing keys fall back to
              _DEFAULTS.  Keys: sigma_m, beta, alpha, gap_s, max_speed_ms,
              cutoff_m, max_obs.

    Returns
    -------
    pd.DataFrame with columns  obs_idx (int32), arc_id (int32).
    Exactly one row per unique obs_idx in cands.  arc_id == -1 for
    observations that had no candidates.
    """
    p = {**_DEFAULTS, **(params or {})}
    # Accept both sigma_m (our API) and sigma_z (pipeline.py legacy key)
    sigma_m: float    = float(p.get("sigma_m") or p.get("sigma_z") or _DEFAULTS["sigma_m"])
    beta: float       = float(p["beta"])
    alpha: float      = float(p["alpha"])
    gap_s: float      = float(p["gap_s"])
    max_speed_ms: float = float(p["max_speed_ms"])
    cutoff_m: float   = float(p["cutoff_m"])
    max_obs: int      = int(p["max_obs"])

    has_probs = "p" in cands.columns and alpha > 0.0

    # Clear per-trip shortest-path cache
    clear_cache(network)

    # ── Build auxiliary lookups ───────────────────────────────────────────
    # All unique observations (some may have 0 candidates)
    # cands covers only obs that have ≥1 candidate.
    obs_with_cands = cands["obs_idx"].unique()

    # Candidate map: obs_idx → sorted list of arc_ids
    cand_map: dict[int, list[int]] = {}
    for oi, grp in cands.groupby("obs_idx", sort=False):
        cand_map[int(oi)] = grp["arc_id"].tolist()

    # Timestamps: use first candidate row's t per obs_idx
    t_map = (
        cands.groupby("obs_idx")["t"].first().to_dict()
    )

    # Spatial coords for GPS-distance calculation
    lon_col = "lon" if "lon" in cands.columns else None
    lat_col = "lat" if "lat" in cands.columns else None
    if lon_col:
        lon_map = cands.groupby("obs_idx")[lon_col].first().to_dict()
        lat_map = cands.groupby("obs_idx")[lat_col].first().to_dict()
    else:
        lon_map = {oi: 0.0 for oi in t_map}
        lat_map = {oi: 0.0 for oi in t_map}

    # Build a fake obs_df for _split_segments
    obs_df = pd.DataFrame(
        {
            "obs_idx": sorted(t_map.keys()),
            "t": [t_map[i] for i in sorted(t_map.keys())],
            "lon": [lon_map[i] for i in sorted(t_map.keys())],
            "lat": [lat_map[i] for i in sorted(t_map.keys())],
        }
    ).sort_values("obs_idx").reset_index(drop=True)

    # ── Log-emission lookup ───────────────────────────────────────────────
    log_em = build_log_emission_matrix(
        cands, sigma_m=sigma_m, alpha=alpha, has_probs=has_probs
    )

    # ── Downsample if too large ───────────────────────────────────────────
    all_obs_idxs = obs_df["obs_idx"].tolist()
    if len(all_obs_idxs) > max_obs:
        stride = math.ceil(len(all_obs_idxs) / max_obs)
        decode_idxs = set(all_obs_idxs[::stride])
        obs_df_decode = obs_df[obs_df["obs_idx"].isin(decode_idxs)].copy()
    else:
        decode_idxs = set(all_obs_idxs)
        obs_df_decode = obs_df

    # ── Split into segments and decode ───────────────────────────────────
    segments = _split_segments(obs_df_decode, gap_s=gap_s, max_speed_ms=max_speed_ms)
    decoded_sparse: dict[int, int] = {}

    n_restarts = 0  # count chain breaks

    for seg in segments:
        seg_obs_idxs = seg["obs_idx"].tolist()
        seg_times = {int(r.obs_idx): float(r.t) for r in seg.itertuples()}
        seg_lons  = {int(r.obs_idx): float(r.lon) for r in seg.itertuples()}
        seg_lats  = {int(r.obs_idx): float(r.lat) for r in seg.itertuples()}

        seg_decoded = _viterbi_segment(
            obs_idxs=seg_obs_idxs,
            cand_map=cand_map,
            log_em=log_em,
            times=seg_times,
            lons=seg_lons,
            lats=seg_lats,
            network=network,
            beta=beta,
            cutoff_m=cutoff_m,
        )
        decoded_sparse.update(seg_decoded)

    # ── Assign all observations (including skipped by downsampling) ───────
    full_decoded = _downsample_and_assign(all_obs_idxs, decoded_sparse, max_obs)

    # Ensure -1 for obs that have no candidates AND weren't decoded
    for oi in all_obs_idxs:
        if oi not in full_decoded:
            full_decoded[oi] = -1

    # ── Build output DataFrame ────────────────────────────────────────────
    out = pd.DataFrame(
        {
            "obs_idx": np.array(all_obs_idxs, dtype=np.int32),
            "arc_id": np.array([full_decoded[oi] for oi in all_obs_idxs], dtype=np.int32),
        }
    )
    return out
