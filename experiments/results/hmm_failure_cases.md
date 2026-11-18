# HMM Failure Cases

This file documents trips and segments where the HMM either did not improve over the
classifier-only baseline or made the prediction worse.  Identified from
`experiments/results/hmm_comparison.csv`.

---

## Known failure-prone scenarios (pre-analysis)

### 1. Long GPS gaps (Δt > 60 s)
**Trips:** 90 (153 064 s gap), 78 (52 800 s gap), 69 (2 660 s gap), 21 (2 160 s gap)

The Viterbi chain is broken at gap boundaries and each segment is decoded
independently.  For very long gaps the two segments may reconstruct
disconnected routes, so route continuity drops sharply at the break point.
Classifier-only is equally affected, but HMM cannot "bridge" the gap.

### 2. U-turns and parallel roads
When the GPS trace is on a road with a close parallel twin (e.g. divided
highway), the Gaussian emission gives nearly identical log-probability to
both arcs.  The transition model prefers the arc that minimises
|d_route − d_GPS|, but on a straight road both have the same route distance,
so the model may flip between the two arcs.  Route continuity suffers.

### 3. Label noise in reference route
The reference `.route` file is 98–100 % connected but not perfect.  Where
the reference itself is wrong or ambiguous (ramps, service roads), both
classifier and HMM produce a "correct" but non-matching arc, inflating
apparent error.

### 4. Very sparse sampling (> 1 s intervals)
Trips 4, 22, 39, 46 (≈55 % 1 s steps) have many steps where the GPS
displacement is large relative to arc lengths.  The Dijkstra cutoff
(cutoff_m = 2000 m) may be too tight, causing LARGE_PENALTY transitions
and chain restarts that the classifier-only method avoids.

### 5. Irregular city grids (many short arcs)
On trips with very small arcs (< 10 m) the same GPS point projects onto
many candidates simultaneously, all with similar dist_m.  The transition
term dominates but provides no useful signal when arcs are all adjacent.
HMM gives marginal benefit here.

---

## Post-experiment analysis
*To be populated after `python scripts/run_hmm_experiments.py` completes.*

| trip_id | issue | hmm_f1 | clf_f1 | Δf1 | notes |
|---------|-------|--------|--------|-----|-------|
| (TBD)   |       |        |        |     |       |

Map screenshots for failure cases: `reports/maps/failure_*.html`
