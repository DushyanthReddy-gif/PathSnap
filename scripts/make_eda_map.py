"""EDA Folium map for ONE trip: GPS points + nearby road network + reference route.
Run from the repo root:   python scripts/make_eda_map.py --trip 00000052
Writes reports/maps/eda_map_<trip>.html   (open it in a browser).
Needs:  pip install folium
EDA only: uses train / student_test trips (it reads the reference route).
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import folium

from pathsnap import io
from pathsnap.geo import LocalProjection
from pathsnap.network import RoadNetwork

ap = argparse.ArgumentParser()
ap.add_argument("--trip", default="00000052")
ap.add_argument("--max_gps", type=int, default=2000)       # draw at most this many GPS points
ap.add_argument("--max_segments", type=int, default=3000)  # draw at most this many network roads
ap.add_argument("--near_m", type=float, default=100.0)     # show roads within this distance of the GPS path
args = ap.parse_args()

d = next((ROOT / "data" / "raw" / s / args.trip for s in ("student_test", "train")
          if (ROOT / "data" / "raw" / s / args.trip).exists()), None)
if d is None:
    sys.exit(f"trip {args.trip} not found in data/raw/train or data/raw/student_test")

trk = io.load_track(d)
nodes, arcs = io.load_network(d)
route = io.load_route(d)
net = RoadNetwork(nodes, arcs, LocalProjection.from_track(trk))
px, py = net.proj.to_xy(trk["lon"].to_numpy(), trk["lat"].to_numpy())

# Folium wants [lat, lon]; the raw files are lon, lat
def arc_latlon(a):
    f, t = arcs["from_node"].to_numpy()[a], arcs["to_node"].to_numpy()[a]
    return [[nodes["lat"].iat[f], nodes["lon"].iat[f]], [nodes["lat"].iat[t], nodes["lon"].iat[t]]]

# roads near the GPS path (every 5th point is enough to find them)
_, near_arcs, _, _ = net.candidates_near(px[::5], py[::5], args.near_m)
near_arcs = np.unique(near_arcs)
if len(near_arcs) > args.max_segments:
    near_arcs = np.random.default_rng(42).choice(near_arcs, args.max_segments, replace=False)

lat_c, lon_c = trk["lat"].mean(), trk["lon"].mean()
# Esri light gray loads from file://; OpenStreetMap refuses tiles without a Referer (HTTP 403)
# and CARTO serves "API key required" tiles, so neither is used as the default background.
m = folium.Map(location=[lat_c, lon_c], zoom_start=14, tiles=None)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors",
    name="Esri light gray", max_native_zoom=16, max_zoom=19,
).add_to(m)

g_net = folium.FeatureGroup(name="Road network (near trip)")
for a in near_arcs:
    folium.PolyLine(arc_latlon(a), color="#888888", weight=2, opacity=0.6).add_to(g_net)
g_net.add_to(m)

g_ref = folium.FeatureGroup(name="Reference route")
for a in route:
    folium.PolyLine(arc_latlon(a), color="#1a9850", weight=5, opacity=0.8).add_to(g_ref)
g_ref.add_to(m)

g_gps = folium.FeatureGroup(name="Raw GPS points")
step = max(1, len(trk) // args.max_gps)
for i in range(0, len(trk), step):
    folium.CircleMarker([trk["lat"].iat[i], trk["lon"].iat[i]], radius=2, color="#2166ac", fill=True,
                        fill_opacity=0.9, tooltip=f"obs {i}, t={trk['t'].iat[i]:.0f}s").add_to(g_gps)
g_gps.add_to(m)

m.fit_bounds([[trk["lat"].min(), trk["lon"].min()], [trk["lat"].max(), trk["lon"].max()]])
folium.LayerControl(collapsed=False).add_to(m)
legend = ("<div style='position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 10px;"
          "border:1px solid #999;font-size:13px'><b>Trip " + args.trip + "</b><br>"
          "<span style='color:#2166ac'>&#9679;</span> raw GPS points<br>"
          "<span style='color:#1a9850'>&#9644;</span> reference route<br>"
          "<span style='color:#888888'>&#9644;</span> road network (near trip)</div>")
m.get_root().html.add_child(folium.Element(legend))

out = ROOT / "reports" / "maps"; out.mkdir(parents=True, exist_ok=True)
path = out / f"eda_map_{args.trip}.html"
m.save(str(path))
print(f"saved {path}  | GPS drawn: {len(range(0, len(trk), step))}/{len(trk)} | network roads: {len(near_arcs)} | route roads: {len(route)}")
