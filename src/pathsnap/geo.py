"""Geometry helpers. All distances are in METRES in a local flat (x, y) plane.

Why not use degrees directly?  One degree of longitude is a different number of
metres at different latitudes (about 111 km at the equator but only about 55 km at
60 degrees north), so "Euclidean distance in degrees" is wrong.

Projection used (equirectangular, centred on each trip):
    x = R * cos(lat0) * (lon - lon0)        y = R * (lat - lat0)
with lon, lat in radians and R = 6,371,000 m.  Over one city (a few tens of km)
the error is tiny, it needs no extra library, and it works anywhere on Earth
because we re-centre on every trip.
"""
import numpy as np

EARTH_R = 6_371_000.0


class LocalProjection:
    def __init__(self, lon0, lat0):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self._cos = np.cos(np.radians(self.lat0))

    @classmethod
    def from_track(cls, track_df):
        """Centre the projection on the mean GPS position of the trip."""
        return cls(track_df["lon"].mean(), track_df["lat"].mean())

    def to_xy(self, lon, lat):
        lon = np.asarray(lon, dtype="float64")
        lat = np.asarray(lat, dtype="float64")
        x = EARTH_R * self._cos * np.radians(lon - self.lon0)
        y = EARTH_R * np.radians(lat - self.lat0)
        return x, y

    def to_lonlat(self, x, y):
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        lon = self.lon0 + np.degrees(x / (EARTH_R * self._cos))
        lat = self.lat0 + np.degrees(y / EARTH_R)
        return lon, lat


def bearing_deg(x1, y1, x2, y2):
    """Compass bearing of the vector (x1,y1)->(x2,y2): 0 = north, 90 = east, range [0, 360)."""
    return np.degrees(np.arctan2(np.asarray(x2) - x1, np.asarray(y2) - y1)) % 360.0


def angle_diff_deg(a, b):
    """Smallest difference between two bearings, in [0, 180]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)


def point_to_segment(px, py, x1, y1, x2, y2):
    """Distance from points to segments (all arrays of equal length, or broadcastable).

    Returns (dist_m, frac) where frac in [0, 1] is how far along the segment
    (0 = start, 1 = end) the closest point lies.
    """
    dx, dy = x2 - x1, y2 - y1
    seg2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = ((px - x1) * dx + (py - y1) * dy) / seg2
    frac = np.where(seg2 > 0, frac, 0.0)          # zero-length segment: use its start
    frac = np.clip(frac, 0.0, 1.0)
    cx, cy = x1 + frac * dx, y1 + frac * dy
    return np.hypot(px - cx, py - cy), frac
