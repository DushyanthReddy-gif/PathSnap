# PathSnap — Team Plan (read this first)

**Team:** Poojitha (P1) · Aditya (P2) · Leka (P3) · Dushyanth (P4)
**Hard stop:** 15:30–16:00 IST today · **Phase-1 evaluation:** before lunch (exact time TBD — assume ~12:30)
**Primary source:** `Main Guidelines.pdf` §4 (12-stage work process). Every stage below maps to it.

| Person | Role | Guideline stages | Report sections | Laptop |
|---|---|---|---|---|
| **P1 Poojitha** | Data, labels, candidates, features, EDA · **report editor** | 2, 3, 12 | §2.2 + assembles PDF | own |
| **P2 Aditya** | Model comparison, Bayesian optimisation, model export | 4, 5, 6, 8 | §3, §4 | **Laptop 1 (runs the API)** |
| **P3 Leka** | HMM/Viterbi, route metrics, all Folium maps | 7 | §5, §6 | own |
| **P4 Dushyanth** | Repo owner, pipeline, FastAPI, Streamlit UI, deployment | 9, 10, 11 | §7 | **Laptop 2 (runs the UI)** |

---

## 1. Facts about the data everyone must know

Verified by profiling all 90 trips in `Student folder/`:

- **90 trips:** 80 `train` + 10 `student_test`. The 10 evaluator trips (IDs 15, 17, 23, 30, 49, 59, 67, 73, 87, 94) are **not** given; they will be revealed later. The system must load them from a folder in the same format.
- **Files per trip:**

  | File | Contents |
  |---|---|
  | `.track` | `lon lat t` |
  | `.nodes` | `lon lat`; the row index is `node_id` |
  | `.arcs` | `from_node to_node`; the row index is `arc_id` |
  | `.route` | ordered `arc_id`s |

  None of the files has a header. **The column order is lon, lat. Folium wants [lat, lon].** This is the #1 bug to avoid.
- **Each trip has its own road network**, and the trips are worldwide (Germany, Russia, USA, Vietnam, Japan, Argentina, …). All distances must be in metres in a **local projection per trip**. Never use degree-Euclidean distance.
- **Network size varies:**
  - 299 to 155k nodes, and 649 to 307k arcs per trip.
  - Arcs are **directed**: 82–99% have a reverse twin. Direction matters because `.route` uses directed arc IDs.
- **GPS points per trip range from 262 to 19,313** (median ≈1,500; 227k in total). Sampling is mostly 1 Hz, with exceptions:
  - Trips 4, 22, 39 and 46 are only ~55% 1 s steps.
  - Big gaps: trip 90 has a 153,064 s gap, trip 78 has 52,800 s, trip 69 has 2,660 s, and trip 21 has 2,160 s.
- **`.route` is an ordered list of arcs, NOT one label per GPS point.** Per-observation labels must be *derived* by P1; this defines Top-1 accuracy for everyone. Routes are 98–100% connected, so the reference has small imperfections (mention this in the report).
- **No NaNs** in any file.
