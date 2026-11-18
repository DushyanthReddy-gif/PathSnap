"""Features for each (GPS point, candidate road) pair.  Never reads the reference route.

Each feature answers "how likely is this road the right one?":
  dist_m            distance from the GPS point to the road (closer = more likely)
  dist_rank         0 = nearest candidate of this point, 1 = second nearest ...
  dist_delta        dist_m minus the nearest candidate's dist_m (0 for the nearest)
  n_cands           how many candidates this GPS point has (crowded junction or not)
  proj_frac         where along the road the point lands (0 = start, 1 = end)
  arc_length_m      road length (very short roads are junction pieces)
  heading_diff_deg  angle between the car's direction of travel and the road direction
                    (0 = same way, 180 = opposite). Separates a road from its reverse twin.
  speed_mps         car speed from neighbouring GPS points (metres/second)
  dt_gap_s          longest time gap to the neighbouring GPS points (sparse or gappy data)
  deg_to_out        number of roads leaving the end of this road (junction complexity)
Group-relative features (rank, delta, n_cands) are computed within each GPS point.
"""
import numpy as np

from .geo import angle_diff_deg, bearing_deg

FEATURE_COLUMNS = ["dist_m", "dist_rank", "dist_delta", "n_cands", "proj_frac", "arc_length_m",
                   "heading_diff_deg", "speed_mps", "dt_gap_s", "deg_to_out"]

MIN_MOVE_M = 3.0      # below this the heading is unreliable (car almost still)
MAX_HEADING_DT_S = 10.0
MAX_SPEED_MPS = 60.0
MAX_DT_GAP_S = 60.0


def _motion(track, proj):
    """Per-observation heading (deg), speed, dt gap and heading validity from neighbouring points."""
    x, y = proj.to_xy(track["lon"].to_numpy(), track["lat"].to_numpy())
    t = track["t"].to_numpy()
    n = len(t)
    i = np.arange(n)
    p, q = np.clip(i - 1, 0, n - 1), np.clip(i + 1, 0, n - 1)
    dx, dy, dtt = x[q] - x[p], y[q] - y[p], t[q] - t[p]
    move = np.hypot(dx, dy)
    heading = bearing_deg(0.0, 0.0, dx, dy)
    valid = (move >= MIN_MOVE_M) & (dtt > 0) & (dtt <= MAX_HEADING_DT_S)
    speed = np.where(dtt > 0, move / np.maximum(dtt, 1e-6), 0.0).clip(0, MAX_SPEED_MPS)
    dt_gap = np.maximum(t[i] - t[p], t[q] - t[i]).clip(0, MAX_DT_GAP_S)
    return heading, valid, speed, dt_gap


def build_features(cands, track, network):
    """cands: output of generate_candidates. Returns a copy with FEATURE_COLUMNS added (float32)."""
    df = cands.copy()
    g = df.groupby("obs_idx")["dist_m"]
    df["dist_rank"] = g.rank(method="first").to_numpy() - 1
    df["dist_delta"] = df["dist_m"] - g.transform("min")
    df["n_cands"] = g.transform("size")

    arc = df["arc_id"].to_numpy()
    df["arc_length_m"] = network.length_m[arc]
    out_deg = np.bincount(network.from_node, minlength=network.n_nodes)
    df["deg_to_out"] = out_deg[network.to_node[arc]]

    heading, valid, speed, dt_gap = _motion(track, network.proj)
    o = df["obs_idx"].to_numpy()
    diff = angle_diff_deg(heading[o], network.bearing_deg[arc])
    df["heading_diff_deg"] = np.where(valid[o], diff, 90.0)     # 90 = "no information"
    df["speed_mps"] = speed[o]
    df["dt_gap_s"] = dt_gap[o]

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype("float32")
    return df
