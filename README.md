# PathSnap 🛰️🛣️

> Robust, high-accuracy map-matching and trajectory reconstruction pipeline combining candidate generation, machine learning classifiers, and Hidden Markov Models (HMM / Viterbi decoding).

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Overview & Architecture

PathSnap reconstructs the true sequence of road network arcs traversed by noisy GPS observation sequences across worldwide road networks.

### Distributed Deployment Architecture (Two Laptops over Hotspot)

```
┌─────────────────────────────────────────────────────────────┐
│                   LAPTOP 1 (Aditya / Fedora)                │
│                                                             │
│   FastAPI Backend Service (:8000)                           │
│   ├── Loads model artifacts:                                │
│   │   ├── artifacts/baseline_model.joblib                   │
│   │   ├── artifacts/final_model.joblib                      │
│   │   ├── artifacts/model_meta.json                         │
│   │   └── artifacts/hmm_params.json                         │
│   ├── Endpoints:                                            │
│   │   ├── GET  /health                                      │
│   │   ├── GET  /trips                                       │
│   │   └── POST /predict                                     │
│   └── Live End-to-End Pipeline:                             │
│       Raw Data (trips_dir) ──> Spatial Candidates           │
│       ──> Feature Matrix ──> ML Classifier Probabilities    │
│       ──> HMM Viterbi Decoder ──> Reconstructed Route       │
│       ──> Server-side Metrics Evaluation (via .route)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                HTTP / JSON    │ Local WiFi Hotspot
                Port 8000      │ (e.g. 192.168.43.x)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 LAPTOP 2 (Dushyanth / Client)               │
│                                                             │
│   Streamlit Evaluator UI (:8501)                            │
│   ├── Holds NO models and NO raw data                       │
│   ├── Connects to Laptop 1 via PATHSNAP_API_URL             │
│   ├── Selects trip ID dynamically from GET /trips           │
│   ├── Displays:                                             │
│   │   ├── Top-1 Accuracy (Classifier vs HMM)                │
│   │   ├── Route Continuity                                  │
│   │   ├── Sequence Precision / Recall / F1                  │
│   │   ├── Candidate Miss Rate                               │
│   │   └── Interactive Folium Map:                           │
│   │       (GPS points, Network context, Predicted route)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Project File Structure

```
pathsnap/
├── README.md                               # Project documentation & instructions
├── requirements.txt                        # Full environment dependencies
├── requirements-ui.txt                     # Lightweight UI dependencies (Laptop 2)
├── pyproject.toml                          # Editable package install configuration
├── .gitignore                              # Git exclusion rules
├── AI_USAGE.md                             # AI usage disclosure (≤ 6 lines)
├── configs/
│   ├── common.yaml                         # Global seed (42), paths, projection
│   ├── candidates.yaml                     # Search radius, max_k, stride
│   ├── models.yaml                         # Hyperparameters for candidate classifiers
│   ├── optuna.yaml                         # Bayesian optimization search space
│   ├── hmm.yaml                            # Transition & emission parameters
│   └── api.yaml                            # API host, port, trips_dir, display limits
├── data/
│   ├── raw/
│   │   ├── train/<trip_id>/                # 80 training trips (.track, .nodes, .arcs, .route)
│   │   └── student_test/<trip_id>/         # 10 test trips (.track, .nodes, .arcs, .route)
│   ├── eval/                               # Evaluator trips directory (for live evaluation)
│   └── processed/                          # Gitignored: generated parquet datasets
├── src/pathsnap/
│   ├── io.py                               # Raw data loaders (track, network, route)
│   ├── geo.py                              # Local projection, metric distances, bearings
│   ├── network.py                          # RoadNetwork representation & spatial indexing
│   ├── candidates.py                       # Observation candidate arc generator
│   ├── labels.py                           # Route alignment & per-observation ground truth
│   ├── features.py                         # Spatial & kinematic feature engineering
│   ├── models/
│   │   ├── registry.py                     # Classifier pipeline factory
│   │   ├── train.py                        # GroupKFold cross-validation & training
│   │   └── tune.py                         # Optuna Bayesian optimization
│   ├── hmm/
│   │   ├── emissions.py                    # Emission probability formulations
│   │   ├── transitions.py                  # Network topology transition penalty matrix
│   │   └── viterbi.py                      # Viterbi dynamic programming decoder
│   ├── evaluation/
│   │   ├── candidate_metrics.py            # Classification metrics & Top-1 accuracy
│   │   └── route_metrics.py                # Route continuity, F1, candidate miss rate
│   ├── viz/
│   │   ├── plots.py                        # Model comparison & training curve plots
│   │   └── maps.py                         # Folium interactive map builder
│   └── pipeline.py                         # Central predict_trip() and score_trip()
├── scripts/
│   ├── run_eda.py                          # Exploratory data analysis
│   ├── build_dataset.py                    # Preprocesses raw trips into parquet tables
│   ├── compare_models.py                   # Benchmarks 5+ candidate classifier models
│   ├── tune_model.py                       # Runs Optuna hyperparameter optimization
│   ├── export_model.py                     # Serializes best model to artifacts/
│   ├── run_hmm_experiments.py              # Evaluates HMM configurations
│   ├── make_maps.py                        # Generates offline HTML map visualizations
│   └── smoke_test_pipeline.py              # End-to-end test verifying all 10 test trips
├── api/
│   ├── main.py                             # FastAPI application & endpoints
│   ├── schemas.py                          # Pydantic request / response schemas
│   └── logging_config.py                   # Structured request logging
├── ui/
│   └── app.py                              # Streamlit evaluation dashboard
├── artifacts/                              # Saved models and meta parameters
│   ├── baseline_model.joblib
│   ├── final_model.joblib
│   ├── model_meta.json
│   └── hmm_params.json
├── experiments/
│   ├── logs/                               # Execution logs
│   └── results/                            # Metrics CSVs and benchmarking summaries
└── reports/
    ├── figures/                            # Exported charts for report
    ├── maps/                               # Exported Folium maps
    └── screenshots/                        # UI and terminal log screenshots
```

---

## 3. Environment Setup

### Full Environment (Laptop 1 / Development)

```bash
# Clone the repository
git clone https://github.com/DushyanthReddy-gif/PathSnap.git
cd PathSnap

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .
```

### UI-Only Environment (Laptop 2)

```bash
git clone https://github.com/DushyanthReddy-gif/PathSnap.git
cd PathSnap
python -m venv .venv
# activate venv
pip install -r requirements-ui.txt
```

---

## 4. Reproducing the Pipeline (Step-by-Step)

Each step has a dedicated standalone script runnable from the repo root:

```bash
# 1. Exploratory Data Analysis
python scripts/run_eda.py

# 2. Build Processed Dataset (generates data/processed/*.parquet)
python scripts/build_dataset.py

# 3. Benchmark Candidate Classification Models (5+ models)
python scripts/compare_models.py

# 4. Bayesian Hyperparameter Optimization (Optuna)
python scripts/tune_model.py

# 5. Export Baseline and Final Models
python scripts/export_model.py

# 6. Run HMM Viterbi Decoder Experiments
python scripts/run_hmm_experiments.py

# 7. Generate Folium Maps
python scripts/make_maps.py

# 8. Run Smoke Test across all 10 student_test trips
python scripts/smoke_test_pipeline.py
```

---

## 5. Deployment Guide

### On Laptop 1 (Aditya / Fedora — Backend API)

1. Open port 8000 on firewall:
   ```bash
   sudo firewall-cmd --add-port=8000/tcp --permanent
   sudo firewall-cmd --reload
   ```
2. Check IP address on hotspot:
   ```bash
   ip -brief address
   # Note the IP, e.g. 192.168.43.100
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

### On Laptop 2 (Dushyanth / Client — Streamlit UI)

1. Set the environment variable and run Streamlit:
   ```bash
   # On Linux / macOS:
   export PATHSNAP_API_URL=http://<LAPTOP1_IP>:8000
   streamlit run ui/app.py

   # On Windows (cmd):
   set PATHSNAP_API_URL=http://<LAPTOP1_IP>:8000
   streamlit run ui/app.py

   # On Windows (PowerShell):
   $env:PATHSNAP_API_URL="http://<LAPTOP1_IP>:8000"
   streamlit run ui/app.py
   ```

### Switching to Evaluator Trips (`data/eval/`)

When the 10 evaluator trips are provided:
1. Copy the unzipped trip folders into `data/eval/` on Laptop 1.
2. In `configs/api.yaml`, change:
   ```yaml
   trips_dir: data/eval
   ```
3. The API will dynamically serve the new trip IDs on the next request.

---

## 6. Team & Responsibilities

| Person | Role | Deliverables & Responsibilities |
|---|---|---|
| **Poojitha (P1)** | Data & Feature Engineering | I/O, local UTM projections, road network graph, candidate generation, ground truth alignment, feature engineering, EDA, report assembly |
| **Aditya (P2)** | Models & Bayesian Optimization | Classifier benchmarking (Logistic Regression, RF, LightGBM, MLP), Optuna hyperparameter tuning, model serialization |
| **Leka (P3)** | HMM / Viterbi & Visualizations | Transition and emission formulations, Viterbi dynamic programming decoder, route continuity and sequence metrics, Folium interactive maps |
| **Dushyanth (P4)** | Repo Owner, Pipeline & Deployment | Git repository owner, unified pipeline (`pipeline.py`), FastAPI backend, Streamlit UI, distributed hotspot deployment, smoke tests |

---

## 7. Results Summary

- Model benchmarking results: `experiments/results/model_comparison.csv`
- Optuna optimization trials: `experiments/results/optuna_trials.csv`
- HMM ablation and tuning: `experiments/results/hmm_experiments.csv`
- Full test evaluation: `experiments/results/student_test_metrics.csv`
