"""
Emission log-probabilities for the HMM decoder.

Two modes
---------
v1  (distance-only)  – Gaussian on perpendicular distance to the arc.
    log p(o | state j) = -0.5 * (dist_m / sigma_m)^2  (+ constant dropped)

final – blend of classifier probability and distance Gaussian:
    log p = alpha * log(p_clf) + (1-alpha) * [-0.5*(dist_m/sigma)^2]

Reference: Newson & Krumm (2009) §3.1 — Gaussian noise model on GPS error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Minimum log-probability floor to avoid -inf from zero classifier probs.
_LOG_FLOOR = -50.0


def gaussian_log_emission(dist_m: "np.ndarray | pd.Series", sigma_m: float) -> np.ndarray:
    """Return log N(dist; 0, sigma) up to an additive constant.

    Parameters
    ----------
    dist_m  : perpendicular distances to candidate arcs (metres).
    sigma_m : GPS noise standard deviation (metres).

    Returns
    -------
    1-D float64 array of log-emission values (always ≤ 0).
    """
    d = np.asarray(dist_m, dtype=np.float64)
    return -0.5 * (d / sigma_m) ** 2


def classifier_log_emission(
    p_clf: "np.ndarray | pd.Series",
    dist_m: "np.ndarray | pd.Series",
    sigma_m: float,
    alpha: float,
) -> np.ndarray:
    """Blend classifier log-probability with distance Gaussian.

    log p = alpha * log(p_clf) + (1-alpha) * log N(dist; 0, sigma)

    Parameters
    ----------
    p_clf   : classifier probability that the candidate is the true arc.
    dist_m  : perpendicular distance (metres).
    sigma_m : GPS noise sigma.
    alpha   : blend weight for classifier term (0 → distance-only, 1 → classifier-only).

    Returns
    -------
    1-D float64 array.
    """
    p = np.asarray(p_clf, dtype=np.float64)
    log_clf = np.where(p > 0, np.log(p), _LOG_FLOOR)
    log_dist = gaussian_log_emission(dist_m, sigma_m)
    return alpha * log_clf + (1.0 - alpha) * log_dist


def build_log_emission_matrix(
    obs_cands: pd.DataFrame,
    sigma_m: float,
    alpha: float = 0.0,
    has_probs: bool = False,
) -> dict[int, dict[int, float]]:
    """Build a nested mapping  obs_idx -> {arc_id -> log_emission}.

    If ``has_probs`` is True the DataFrame must contain a column ``p``.
    When ``alpha == 0`` (default) the classifier column is ignored even if
    present, so v1 (distance-only) works without P2's model.

    Parameters
    ----------
    obs_cands : candidate DataFrame with columns obs_idx, arc_id, dist_m[, p].
    sigma_m   : GPS noise sigma (metres).
    alpha     : blend weight; 0 → pure Gaussian.
    has_probs : whether to read the ``p`` column.

    Returns
    -------
    dict mapping obs_idx → {arc_id: log_emission_value}
    """
    result: dict[int, dict[int, float]] = {}

    for obs_idx, grp in obs_cands.groupby("obs_idx", sort=False):
        arc_ids = grp["arc_id"].to_numpy(dtype=np.int32)
        dist_m = grp["dist_m"].to_numpy(dtype=np.float64)

        if has_probs and alpha > 0.0 and "p" in grp.columns:
            p_clf = grp["p"].to_numpy(dtype=np.float64)
            log_em = classifier_log_emission(p_clf, dist_m, sigma_m, alpha)
        else:
            log_em = gaussian_log_emission(dist_m, sigma_m)

        result[int(obs_idx)] = dict(zip(arc_ids.tolist(), log_em.tolist()))

    return result
