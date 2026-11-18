"""RoadNetwork: the road graph of ONE trip, in metres, with a fast spatial lookup.

Attributes (arrays are numpy, indexed by arc_id / node_id):
    n_nodes, n_arcs
    node_x, node_y           node positions in metres
    from_node, to_node       the two ends of each directed arc
    x1, y1, x2, y2           arc end points in metres
    length_m                 arc length
    bearing_deg              direction of travel along the arc (0 = north, clockwise)
    out_arcs[node]           array of arc_ids that START at node
    in_arcs[node]            array of arc_ids that END at node
    proj                     the LocalProjection used (project more points with proj.to_xy)

Main method:
    candidates_near(px, py, radius_m)  -> (obs_idx, arc_id, dist_m, proj_frac)

How the spatial lookup works: every arc is cut into pieces of about SAMPLE_STEP_M
metres and the middle of each piece goes into a KD-tree.  For a GPS point we ask
the tree for samples within radius + step/2 (so no arc within `radius` can be
missed), then compute the EXACT point-to-segment distance for those few arcs.
"""
import numpy as np
from scipy.spatial import cKDTree

from .geo import bearing_deg, point_to_segment

SAMPLE_STEP_M = 10.0


def _group_by(keys, values, n_groups):
    """out[g] = array of values whose key == g (for g in 0..n_groups-1)."""
    order = np.argsort(keys, kind="stable")
    counts = np.bincount(keys, minlength=n_groups)
    return np.split(values[order], np.cumsum(counts)[:-1])


class RoadNetwork:
    def __init__(self, nodes_df, arcs_df, projection):
        self.proj = projection
        self.node_x, self.node_y = projection.to_xy(nodes_df["lon"].to_numpy(), nodes_df["lat"].to_numpy())
        self.n_nodes = len(nodes_df)
        self.n_arcs = len(arcs_df)

        self.from_node = arcs_df["from_node"].to_numpy()
        self.to_node = arcs_df["to_node"].to_numpy()
        self.x1, self.y1 = self.node_x[self.from_node], self.node_y[self.from_node]
        self.x2, self.y2 = self.node_x[self.to_node], self.node_y[self.to_node]
        self.length_m = np.hypot(self.x2 - self.x1, self.y2 - self.y1)
        self.bearing_deg = bearing_deg(self.x1, self.y1, self.x2, self.y2)

        arc_ids = np.arange(self.n_arcs)
        self.out_arcs = _group_by(self.from_node, arc_ids, self.n_nodes)
        self.in_arcs = _group_by(self.to_node, arc_ids, self.n_nodes)

        self._build_index()

    # ---- spatial index -------------------------------------------------
    def _build_index(self):
        n_pieces = np.maximum(1, np.ceil(self.length_m / SAMPLE_STEP_M)).astype(np.int64)
        arc_of = np.repeat(np.arange(self.n_arcs), n_pieces)
        start = np.cumsum(n_pieces) - n_pieces
        k = np.arange(len(arc_of)) - np.repeat(start, n_pieces)       # piece number inside its arc
        f = (k + 0.5) / np.repeat(n_pieces, n_pieces)                # middle of each piece
        sx = self.x1[arc_of] + f * (self.x2 - self.x1)[arc_of]
        sy = self.y1[arc_of] + f * (self.y2 - self.y1)[arc_of]
        self._sample_arc = arc_of
        self._tree = cKDTree(np.column_stack([sx, sy]))

    def candidates_near(self, px, py, radius_m):
        """All (point, arc) pairs where the arc passes within radius_m of the point.

        px, py: metre coordinates of the GPS points.
        Returns four equal-length arrays: obs_idx, arc_id, dist_m, proj_frac.
        proj_frac (0..1) is where along the arc the point lands (0 = start, 1 = end).
        """
        px = np.asarray(px, dtype="float64")
        py = np.asarray(py, dtype="float64")
        obs_tree = cKDTree(np.column_stack([px, py]))
        hits = obs_tree.sparse_distance_matrix(
            self._tree, radius_m + SAMPLE_STEP_M / 2.0, output_type="ndarray")
        obs = hits["i"].astype(np.int64)
        arc = self._sample_arc[hits["j"]]

        # one row per (obs, arc): many samples of the same arc can be hit
        key = np.unique(obs * np.int64(self.n_arcs) + arc)
        obs, arc = key // self.n_arcs, key % self.n_arcs

        dist, frac = point_to_segment(px[obs], py[obs],
                                      self.x1[arc], self.y1[arc], self.x2[arc], self.y2[arc])
        keep = dist <= radius_m
        return obs[keep], arc[keep], dist[keep], frac[keep]
