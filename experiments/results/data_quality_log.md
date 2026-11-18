# Data-quality log (P1) — partitions: train (80 trips) and student_test (10 trips); evaluator data never touched

| # | Issue found | Evidence (script -> file) | Decision |
|---|---|---|---|
| 1 | No missing values, timestamps sorted in all 90 trips | run_eda.py -> eda_inventory.csv | none needed |
| 2 | Time gaps: 23 trips have a gap > 60 s (train: trip 90 = 153,064 s, trip 78 = 52,810 s, 69, 21, 11); student_test max gap 210 s | eda_inventory.csv | gap-aware features (`dt_gap_s`); decoder splits at gaps (P3) |
| 3 | Reference route is a list of arcs, not one label per GPS point | Dataset Exploration doc | labels derived by monotonic dynamic-programming alignment (labels.py); on 4 test trips it differs from plain nearest-route-arc for 1-25% of points |
| 4 | Reference route does not fit the GPS on some train trips: trip 54 (GPS path 169 km vs route 31 km; 89% of points > 50 m from the route), trip 33 (59%); trips 2, 66, 43 milder (7-12%) | build_dataset.py -> eda_label_quality.csv | keep all rows but flag `label_ok` (point within 50 m of its reference road); model training should use `label_ok == 1`. All 10 student_test trips are clean (0% > 50 m) |
| 5 | Candidate miss rate (radius 50 m, K = 8): 1.27% on trusted train labels (sweep) | candidate_sweep.py -> eda_candidate_sweep.csv | radius 50 m, K = 8 (larger radius gives no gain; K = 12 gains 0.3 pt for 24% more rows) |
| 6 | Train table uses every 3rd GPS point (speed); student_test is full resolution | configs/candidates.yaml | `train_obs_stride: 3` |
| 7 | Reference routes are not perfect ground truth (98-100% of consecutive route arcs connect) | eda_inventory / Dataset doc | all accuracy numbers are "agreement with the reference route" |

## Additional findings (EDA, partitions: train + student_test)
- Projection: equirectangular, centred on each trip's mean position; degree-Euclidean distance is not used. Files store lon, lat; Folium needs lat, lon.
- Missing values: none. Timestamps are sorted in all 90 trips.
- Sampling: 96.7% of steps are exactly 1 s; 70 steps (0.03%) exceed 60 s. Largest gaps: trips 90 (153,064 s), 78 (52,810 s), 69, 21, 11.
- Network: no isolated nodes; 89 of 90 trips have more than one weakly connected component, so the decoder must handle unreachable steps.
- Reference route: at least 98.1% connected in every trip. It is a reference, not verified ground truth.
- Offset trips: in trips 54, 33, 2 and 43 the reference route lies far from the GPS (mean distance 135 m in trip 54, 81 m in trip 33). About 8.7% of train observations have no candidate within 50 m; student_test has none.
- Decision: labels with label_dist_m <= 50 are used as trusted training labels. Evaluation counts every observation.
- Label method: monotonic DP alignment never moves backward along the route; nearest-arc labelling did so 173 times in trip 54. The two agree on 93-99.5% of points in the 4 trips tested (eda_label_method_comparison.csv).
