# P4 — Dushyanth: Repo Owner, Pipeline, FastAPI Backend, Streamlit UI, Deployment

**Guideline stages:** 9 (FastAPI backend), 10 (evaluator UI), 11 (integrate and test). Also stage 8, together with P2 (loading the exported artifacts).
**Report section you write:** §7 (API, UI and Deployment): screenshots of the working UI + screenshots of API logs in the terminal **on Laptop 1**.
**Laptops:** the API runs on **Laptop 1 (Aditya's, Fedora)**. The UI runs on **Laptop 2 (yours)**. Both are on Aditya's hotspot.

Read `00_TEAM_PLAN.md` first (§3 file structure, §4.7 API contract, §5 dependencies).

---

## Files you own
```
README.md  requirements.txt  requirements-ui.txt  pyproject.toml  .gitignore  AI_USAGE.md (create)
configs/api.yaml
src/pathsnap/pipeline.py
api/main.py  api/schemas.py  api/logging_config.py
ui/app.py
scripts/smoke_test_pipeline.py
reports/screenshots/*
```

## What you deliver, to whom, by when

| # | Deliverable | Who needs it | Due |
|---|---|---|---|
| D1 | **Public repo**, full folder skeleton (empty `__init__.py`, `.gitkeep`s), `.gitignore`, `requirements*.txt`, `pyproject.toml`, `data/raw/` pushed, 3 collaborators added | everyone | **11:10** |
| D7 | API (stub response) on Laptop 1 ↔ Streamlit on Laptop 2 over the hotspot | Phase-1 demo | **12:30** |
| – | `pipeline.py` wired to real candidates + baseline model + `decode()` v1 | Phase-1 demo | **12:45** |
| – | Final integration with the tuned model + tuned HMM; dry run on all 10 student_test trips via UI | demo | **15:00** |
| D13 | UI + API terminal screenshots in `reports/screenshots/` | P1 (report) | 15:00 |
| – | README complete; final push; repo verified public | submission | 15:40 |

## What you need from others
- **P2 (Aditya), 11:10:** Laptop 1 venv + `sudo firewall-cmd --add-port=8000/tcp` + Laptop 1's IP on the hotspot.
- **P1 (Poojitha), 12:15:** `generate_candidates()` + `build_features()` usable on a trip folder **without `.route`**.
- **P2, 12:40:** `artifacts/baseline_model.joblib` + `model_meta.json`. **14:15:** `final_model.joblib`.
- **P3 (Leka), 12:00:** route metrics. **12:30:** `decode()` v1. **13:30:** `viz/maps.py`. **14:30:** `hmm_params.json`.
- **Before those arrive, build against stubs** (fake candidates, `p = 1/(1+dist)`, argmax "decoder") so the API/UI are never blocked.
