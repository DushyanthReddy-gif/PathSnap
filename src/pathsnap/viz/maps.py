"""
Folium map visualisation for PathSnap (P3 — Leka).

IMPORTANT: this file imports ONLY folium (+ stdlib).
Laptop 2 (UI) has no shapely / sklearn / pyproj.

Public API
----------
make_trip_map(gps_latlon, network_segments=None, reference_segments=None,
              predicted_segments=None, title="") -> folium.Map

Usage
-----
    from pathsnap.viz.maps import make_trip_map
    m = make_trip_map(gps_latlon=[[lat, lon], ...],
                      predicted_segments=[[[lat, lon], [lat, lon]], ...])
    m.save("reports/maps/trip_00000052.html")
"""

from __future__ import annotations

import colorsys
import math
from typing import Sequence

import folium
from folium.plugins import MeasureControl

# ── colour palette ────────────────────────────────────────────────────────
_GPS_COLOUR        = "#2196F3"   # blue
_NETWORK_COLOUR    = "#9E9E9E"   # grey
_REFERENCE_COLOUR  = "#4CAF50"   # green
_PREDICTED_COLOUR  = "#F44336"   # red
_CLASSIFIER_COLOUR = "#FF9800"   # orange (optional extra layer)


# ── legend HTML ───────────────────────────────────────────────────────────

def _build_legend(layers: dict[str, str]) -> str:
    """Build an HTML legend div from {label: colour} pairs."""
    rows = ""
    for label, colour in layers.items():
        rows += (
            f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
            f'<div style="width:18px;height:18px;background:{colour};'
            f'border-radius:3px;margin-right:6px;flex-shrink:0;"></div>'
            f'<span style="font-size:12px;">{label}</span></div>\n'
        )
    return (
        '<div style="'
        "position:fixed;bottom:30px;right:10px;z-index:1000;"
        "background:white;padding:10px 14px;border-radius:6px;"
        "box-shadow:0 2px 8px rgba(0,0,0,.35);font-family:sans-serif;"
        '">\n'
        f"<b style='font-size:13px;'>Legend</b><br/>\n{rows}</div>"
    )


# ── fit-bounds helper ─────────────────────────────────────────────────────

def _bounds_from_latlon(points: list[list[float]]) -> tuple[list, list]:
    """Return ([min_lat, min_lon], [max_lat, max_lon]) from a list of [lat, lon]."""
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return [min(lats), min(lons)], [max(lats), max(lons)]


def _expand_bounds(b_sw, b_ne, more_points):
    if not more_points:
        return b_sw, b_ne
    flat = []
    for item in more_points:
        if item and isinstance(item[0], (list, tuple)):
            flat.extend(item)
        elif item:
            flat.append(item)
    if not flat:
        return b_sw, b_ne
    lats = [p[0] for p in flat] + [b_sw[0], b_ne[0]]
    lons = [p[1] for p in flat] + [b_sw[1], b_ne[1]]
    return [min(lats), min(lons)], [max(lats), max(lons)]


# ── main public function ──────────────────────────────────────────────────

def make_trip_map(
    gps_latlon: Sequence[Sequence[float]],
    network_segments: "Sequence[Sequence[Sequence[float]]] | None" = None,
    reference_segments: "Sequence[Sequence[Sequence[float]]] | None" = None,
    predicted_segments: "Sequence[Sequence[Sequence[float]]] | None" = None,
    title: str = "",
    classifier_segments: "Sequence[Sequence[Sequence[float]]] | None" = None,
) -> folium.Map:
    """
    Build a Folium map with up to 4 (+ 1 optional) layers.

    Parameters
    ----------
    gps_latlon          : [[lat, lon], …]  GPS track points.
    network_segments    : [[[lat, lon], [lat, lon]], …]  Nearby road arcs
                          (capped at ~3,000 segments).
    reference_segments  : [[[lat, lon], [lat, lon]], …]  Ground-truth route.
                          NEVER passed from the API to the UI (server-side only).
    predicted_segments  : [[[lat, lon], [lat, lon]], …]  HMM-decoded route.
    title               : Map title shown in the tab / HTML <title>.
    classifier_segments : (optional) classifier-only predictions for comparison.

    Returns
    -------
    folium.Map  (call .save(path) or display in notebook).

    Coordinate convention
    ---------------------
    ALL inputs must be **[lat, lon]** (Folium convention).
    Raw .track files store lon, lat — the caller must swap before passing here.
    """
    gps_latlon = [list(p) for p in gps_latlon]

    if not gps_latlon:
        # Fallback centre
        centre = [0.0, 0.0]
    else:
        centre = [
            sum(p[0] for p in gps_latlon) / len(gps_latlon),
            sum(p[1] for p in gps_latlon) / len(gps_latlon),
        ]

    # CartoDB Positron as the default background: it loads when the HTML is opened
    # straight from disk (file://). OpenStreetMap's servers refuse tile requests that
    # carry no Referer (HTTP 403), so OSM is offered as a second, selectable base layer.
    m = folium.Map(
        location=centre,
        zoom_start=14,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap (needs http://)").add_to(m)

    # ── Layer 1: Road network context (grey, thin) ────────────────────────
    legend_entries: dict[str, str] = {}

    if network_segments:
        net_fg = folium.FeatureGroup(name="Road network (context)", show=True)
        # Cap at 3,000 for performance
        segs = list(network_segments)[:3_000]
        for seg in segs:
            if seg and len(seg) >= 2:
                folium.PolyLine(
                    locations=[list(pt) for pt in seg],
                    color=_NETWORK_COLOUR,
                    weight=1.5,
                    opacity=0.5,
                ).add_to(net_fg)
        net_fg.add_to(m)
        legend_entries["Road network"] = _NETWORK_COLOUR

    # ── Layer 2: GPS points (blue circles) ───────────────────────────────
    gps_fg = folium.FeatureGroup(name="GPS track", show=True)
    for i, pt in enumerate(gps_latlon):
        folium.CircleMarker(
            location=pt,
            radius=3,
            color=_GPS_COLOUR,
            fill=True,
            fill_color=_GPS_COLOUR,
            fill_opacity=0.7,
            popup=folium.Popup(f"GPS obs #{i}", max_width=120),
        ).add_to(gps_fg)
    gps_fg.add_to(m)
    legend_entries["GPS points"] = _GPS_COLOUR

    # ── Layer 3: Reference route (green) — server-side only ───────────────
    if reference_segments:
        ref_fg = folium.FeatureGroup(name="Reference route (ground truth)", show=True)
        for seg in reference_segments:
            if seg and len(seg) >= 2:
                folium.PolyLine(
                    locations=[list(pt) for pt in seg],
                    color=_REFERENCE_COLOUR,
                    weight=4,
                    opacity=0.85,
                ).add_to(ref_fg)
        ref_fg.add_to(m)
        legend_entries["Reference route"] = _REFERENCE_COLOUR

    # ── Layer 4: Predicted route (red) ────────────────────────────────────
    if predicted_segments:
        pred_fg = folium.FeatureGroup(name="HMM predicted route", show=True)
        for seg in predicted_segments:
            if seg and len(seg) >= 2:
                folium.PolyLine(
                    locations=[list(pt) for pt in seg],
                    color=_PREDICTED_COLOUR,
                    weight=4,
                    opacity=0.9,
                ).add_to(pred_fg)
        pred_fg.add_to(m)
        legend_entries["HMM predicted"] = _PREDICTED_COLOUR

    # ── Optional Layer 5: Classifier-only (orange) ────────────────────────
    if classifier_segments:
        clf_fg = folium.FeatureGroup(name="Classifier-only route", show=False)
        for seg in classifier_segments:
            if seg and len(seg) >= 2:
                folium.PolyLine(
                    locations=[list(pt) for pt in seg],
                    color=_CLASSIFIER_COLOUR,
                    weight=3,
                    opacity=0.8,
                    dash_array="8 4",
                ).add_to(clf_fg)
        clf_fg.add_to(m)
        legend_entries["Classifier-only"] = _CLASSIFIER_COLOUR

    # ── Controls ──────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)
    MeasureControl(primary_length_unit="meters").add_to(m)

    # ── Fit bounds ────────────────────────────────────────────────────────
    if gps_latlon:
        sw, ne = _bounds_from_latlon(gps_latlon)
        # Expand for predicted route
        extra = []
        for layer in [network_segments, predicted_segments, reference_segments]:
            if layer:
                for seg in layer:
                    if seg:
                        extra.extend(seg)
        sw, ne = _expand_bounds(sw, ne, extra)
        m.fit_bounds([sw, ne])

    # ── Legend overlay ────────────────────────────────────────────────────
    if legend_entries:
        legend_html = _build_legend(legend_entries)
        m.get_root().html.add_child(folium.Element(legend_html))

    # ── Title ─────────────────────────────────────────────────────────────
    if title:
        title_html = (
            f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
            f'z-index:1000;background:rgba(255,255,255,.88);padding:6px 16px;'
            f'border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,.3);'
            f'font-family:sans-serif;font-size:14px;font-weight:bold;">'
            f'{title}</div>'
        )
        m.get_root().html.add_child(folium.Element(title_html))

    return m


# ── helper: arc geometry → [[lat, lon], [lat, lon]] ──────────────────────

def arc_to_latlon_segment(
    arc_id: int,
    nodes_df,
    arcs_df,
) -> "list[list[float]] | None":
    """
    Convert one arc into a [[lat, lon], [lat, lon]] pair.

    Parameters
    ----------
    arc_id   : index into arcs_df.
    nodes_df : DataFrame with columns lon, lat (row index = node_id).
    arcs_df  : DataFrame with columns from_node, to_node.

    Returns
    -------
    [[lat_start, lon_start], [lat_end, lon_end]]  or None if out of range.

    Notes
    -----
    Raw files store lon, lat — this function returns [lat, lon] for Folium.
    """
    try:
        arc = arcs_df.iloc[arc_id]
        fn, tn = int(arc["from_node"]), int(arc["to_node"])
        fn_row = nodes_df.iloc[fn]
        tn_row = nodes_df.iloc[tn]
        return [
            [float(fn_row["lat"]), float(fn_row["lon"])],
            [float(tn_row["lat"]), float(tn_row["lon"])],
        ]
    except (IndexError, KeyError):
        return None


def build_arc_segments(
    arc_ids: "list[int] | np.ndarray",
    nodes_df,
    arcs_df,
    max_segs: int = 3_000,
) -> list[list[list[float]]]:
    """
    Convert a list of arc IDs into Folium-ready [[lat,lon],[lat,lon]] segments.

    Parameters
    ----------
    arc_ids  : ordered list of arc IDs to convert.
    nodes_df : DataFrame lon, lat (index = node_id).
    arcs_df  : DataFrame from_node, to_node (index = arc_id).
    max_segs : cap for performance (UI safety).

    Returns
    -------
    List of [[lat, lon], [lat, lon]] pairs.
    """
    segs = []
    for arc_id in arc_ids:
        if arc_id < 0:
            continue
        seg = arc_to_latlon_segment(arc_id, nodes_df, arcs_df)
        if seg:
            segs.append(seg)
        if len(segs) >= max_segs:
            break
    return segs
