"""Streamlit Evaluator UI for PathSnap.

Runs on Laptop 2 (Client). Holds NO model and NO raw data.
Communicates strictly via HTTP with Laptop 1 (FastAPI backend).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

# Try to import team map builder if available, otherwise use local fallback
try:
    from pathsnap.viz.maps import make_trip_map  # type: ignore
except Exception:
    def make_trip_map(
        gps: List[List[float]],
        network_context: Optional[List[List[List[float]]]] = None,
        reference_segments: Optional[List[List[List[float]]]] = None,
        predicted_route: Optional[List[List[List[float]]]] = None,
    ) -> folium.Map:
        """Fallback Folium map builder for trajectory and network segments.

        Coordinates MUST be in [lat, lon] order.
        """
        if not gps:
            return folium.Map(location=[0, 0], zoom_start=2)

        # Center map on mean of GPS points
        avg_lat = sum(p[0] for p in gps) / len(gps)
        avg_lon = sum(p[1] for p in gps) / len(gps)
        fmap = folium.Map(location=[avg_lat, avg_lon], zoom_start=14, tiles="CartoDB positron")

        # 1. Network Context Layer (subtle grey lines)
        if network_context:
            net_group = folium.FeatureGroup(name="Road Network Context", show=True)
            for seg in network_context:
                folium.PolyLine(
                    locations=seg,
                    color="#9E9E9E",
                    weight=1.5,
                    opacity=0.6,
                ).add_to(net_group)
            net_group.add_to(fmap)

        # 2. Predicted Route Layer (bold blue line)
        if predicted_route:
            pred_group = folium.FeatureGroup(name="Predicted Route (HMM)", show=True)
            for seg in predicted_route:
                folium.PolyLine(
                    locations=seg,
                    color="#1976D2",
                    weight=4.0,
                    opacity=0.85,
                ).add_to(pred_group)
            pred_group.add_to(fmap)

        # 3. Reference Route Layer (only if explicitly passed, usually None in evaluator UI)
        if reference_segments:
            ref_group = folium.FeatureGroup(name="Reference Route", show=False)
            for seg in reference_segments:
                folium.PolyLine(
                    locations=seg,
                    color="#4CAF50",
                    weight=3.0,
                    opacity=0.7,
                    dash_array="5, 5",
                ).add_to(ref_group)
            ref_group.add_to(fmap)

        # 4. GPS Observations Layer (orange markers/polyline)
        gps_group = folium.FeatureGroup(name="GPS Observations", show=True)
        # Polyline for trajectory
        folium.PolyLine(
            locations=gps,
            color="#FF5722",
            weight=2.0,
            opacity=0.7,
            dash_array="4, 6",
        ).add_to(gps_group)

        # Start and End Markers
        if len(gps) > 0:
            folium.CircleMarker(
                location=gps[0],
                radius=6,
                color="#4CAF50",
                fill=True,
                fill_color="#4CAF50",
                fill_opacity=0.9,
                popup="Start",
            ).add_to(gps_group)

            folium.CircleMarker(
                location=gps[-1],
                radius=6,
                color="#D32F2F",
                fill=True,
                fill_color="#D32F2F",
                fill_opacity=0.9,
                popup="End",
            ).add_to(gps_group)

        gps_group.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        return fmap


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PathSnap — Trajectory Reconstruction",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛣️ PathSnap: Live Map-Matching & Trajectory Reconstruction")
st.caption("Distributed Evaluation Dashboard — Communicating with FastAPI Backend on Laptop 1")

# ---------------------------------------------------------------------------
# Sidebar: Connection & Health Check
# ---------------------------------------------------------------------------
st.sidebar.header("📡 Backend Connection")

default_api_url = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000").rstrip("/")
api_url = st.sidebar.text_input("FastAPI Base URL", value=default_api_url)

# Health Check Button / Auto-check
health_status = "Unknown"
loaded_model = "None"
is_healthy = False

try:
    health_resp = requests.get(f"{api_url}/health", timeout=5)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        health_status = health_data.get("status", "ok")
        loaded_model = health_data.get("model", "unknown")
        is_healthy = True
    else:
        health_status = f"HTTP {health_resp.status_code}"
except requests.exceptions.RequestException as e:
    health_status = "Unreachable"

if is_healthy:
    st.sidebar.success(f"🟢 Connected to Backend\n\n**Status:** {health_status}\n\n**Model:** `{loaded_model}`")
else:
    st.sidebar.error(f"🔴 Backend Offline\n\n**Status:** {health_status}\n\nCheck IP/Port & firewall on Laptop 1.")

if st.sidebar.button("🔄 Refresh Connection"):
    st.rerun()

# ---------------------------------------------------------------------------
# Trip Selection
# ---------------------------------------------------------------------------
trips: List[str] = []
if is_healthy:
    try:
        trips_resp = requests.get(f"{api_url}/trips", timeout=10)
        if trips_resp.status_code == 200:
            trips = trips_resp.json().get("trips", [])
    except Exception as e:
        st.warning(f"Could not fetch trip list: {e}")

if not trips:
    st.info("No trips loaded. Ensure the FastAPI backend is running and `trips_dir` contains trip folders.")
    st.stop()

col_select, col_btn = st.columns([3, 1])
with col_select:
    selected_trip = st.selectbox("Select Trip ID", options=trips, help="Trips dynamically discovered from backend")

with col_btn:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 Run Reconstruction", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Reconstruction & Metrics Display
# ---------------------------------------------------------------------------
if run_btn and selected_trip:
    with st.spinner(f"Requesting trajectory reconstruction for trip {selected_trip} from {api_url}..."):
        try:
            start_req = time.perf_counter()
            resp = requests.post(
                f"{api_url}/predict",
                json={"trip_id": selected_trip},
                timeout=180,  # 3-minute timeout as specified
            )
            req_latency = time.perf_counter() - start_req

            if resp.status_code == 200:
                result = resp.json()
                st.session_state["last_result"] = result
                st.session_state["req_latency"] = req_latency
            elif resp.status_code == 404:
                st.error(f"Trip '{selected_trip}' was not found on the backend (404 Not Found).")
                st.stop()
            else:
                st.error(f"Backend returned an error: HTTP {resp.status_code}\n\n{resp.text}")
                st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out after 180 seconds. The server took too long to respond.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to {api_url}. Verify Laptop 1 IP and port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.stop()

if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    metrics = res.get("metrics", {})
    timings = res.get("timings", {})
    gps = res.get("gps", [])
    predicted_route = res.get("predicted_route", [])
    network_context = res.get("network_context", [])

    st.markdown("---")
    st.subheader(f"📊 Results for Trip: `{res.get('trip_id')}` ({res.get('n_observations')} Observations)")

    # Metric Cards
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    with m_col1:
        top1_hmm = metrics.get("top1_accuracy_hmm")
        top1_cls = metrics.get("top1_accuracy_classifier_only")
        if top1_hmm is not None:
            delta = f"+{(top1_hmm - top1_cls):.3f} vs Cls" if top1_cls is not None else None
            st.metric("Top-1 Accuracy (HMM)", f"{top1_hmm * 100:.1f}%", delta=delta)
        else:
            st.metric("Top-1 Accuracy", "N/A (No .route)")

    with m_col2:
        continuity = metrics.get("route_continuity")
        if continuity is not None:
            st.metric("Route Continuity", f"{continuity * 100:.1f}%")
        else:
            st.metric("Route Continuity", "N/A")

    with m_col3:
        f1_score = metrics.get("route_f1")
        if f1_score is not None:
            st.metric("Route F1 Score", f"{f1_score:.3f}")
        else:
            st.metric("Route F1 Score", "N/A")

    with m_col4:
        miss_rate = metrics.get("candidate_miss_rate")
        if miss_rate is not None:
            st.metric("Candidate Miss Rate", f"{miss_rate * 100:.2f}%")
        else:
            st.metric("Candidate Miss Rate", "N/A")

    with m_col5:
        total_time = timings.get("total_s", 0.0)
        st.metric("Total Latency", f"{total_time:.2f} s")

    # Map Rendering
    st.markdown("### 🗺️ Trajectory & Map-Matched Route")
    # CRITICAL: Reference route is NOT sent by API and MUST be None
    fmap = make_trip_map(
        gps=gps,
        network_context=network_context,
        reference_segments=None,
        predicted_route=predicted_route,
    )
    st_folium(fmap, width="100%", height=550)

    # Latency Breakdown
    with st.expander("⏱️ Detailed Latency Breakdown"):
        t_c1, t_c2, t_c3, t_c4, t_c5 = st.columns(5)
        t_c1.metric("File I/O", f"{timings.get('load_s', 0.0):.3f} s")
        t_c2.metric("Candidates", f"{timings.get('candidates_s', 0.0):.3f} s")
        t_c3.metric("Classifier", f"{timings.get('model_s', 0.0):.3f} s")
        t_c4.metric("Viterbi HMM", f"{timings.get('viterbi_s', 0.0):.3f} s")
        t_c5.metric("Total Server Time", f"{timings.get('total_s', 0.0):.3f} s")

    # Raw JSON Response
    with st.expander("🔍 Raw JSON Response from API"):
        st.json(res)
