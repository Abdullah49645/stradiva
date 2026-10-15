# Week 1 Feasibility Assessment

**Scope:** prove the core reconstruction chain — line detection → vanishing
point → scale-anchored ground-plane homography → vehicle tracking →
trajectory/speed reconstruction — works end to end, and that the
reconstructability gate correctly refuses unusable footage. Per the brief,
no UI or AWS work has started; this is pure CV/math validation against a
controlled synthetic scene with known ground truth
(`evaluation/dataset/synthetic_scene_A.mp4` + `..._ground_truth.json`).

Scene: elevated camera (h=6 m, pitch=22°), straight single-lane road with
solid edge lines + a standard dashed centerline (3 m dash / 9 m gap),
vehicle moving at a constant, known 20.0 m/s (~72 km/h) for ~4.9 s / 147
frames at 30 fps. Full run: `evaluation/scripts/run_feasibility.py`
(numbers below are its actual output, not projected).

## What works

- **Video ingestion** — clean read of fps/resolution/frame count.
- **Line detection** (Canny + probabilistic Hough, `cv/lines/`) — reliably
  finds both lane edges (7 left / 5 right segments from one frame) and
  rejects near-horizontal clutter via the slope/side clustering filter.
- **Vanishing point** — intersection of the two robustly-fit edge lines
  lands at (635.0, 0.27) px, i.e. essentially the true image center-column,
  which is what a zero-roll, straight-road camera should produce. A crude
  bootstrap-resample stability check gives ±4.2 px std — usable as a first
  confidence signal.
- **Scale recovery from standardized markings** — detecting the dashed
  centerline and using the *assumed* standard dash pattern (3 m/9 m) plus
  *assumed* standard lane half-width (1.85 m) as scale priors, combined
  with the geometric fact that image row depends only on world forward-
  distance (not lateral offset) for a zero-roll camera, yields 36
  image↔world correspondence points from 10 detected dashes and a stable
  homography (69% RANSAC inlier fraction on the first attempt, no manual
  tuning of the fit itself).
- **Ground-contact tracking** — the color-blob tracker held the vehicle
  for all 147 frames, 0 lost.
- **Speed reconstruction is good**: mean estimated speed 19.33 m/s vs
  20.0 m/s true — **3.3% error**, MAE 0.68 m/s, RMSE 0.88 m/s, using only
  a Savitzky-Golay smoother and central-difference velocity. Reconstructed
  and ground-truth position curves are visually parallel
  (`docs/week1_feasibility_plot.png`), confirming the *rate* of motion is
  recovered correctly, not just approximately right on average.
- **The rejection path works.** Feeding the pipeline a deliberately
  degraded frame (heavy blur + flattened contrast, simulating fog/severe
  compression) produces zero usable line segments, so line fitting fails
  and the pipeline correctly reports `NOT_RECONSTRUCTABLE` instead of
  guessing. This is the central product principle (section 4/16 of the
  brief) and it already holds, not just as a hardcoded case.

## What doesn't (yet)

- **Absolute position has a large, systematic bias**: 27.45 m MAE, but
  RMSE (27.47 m) ≈ MAE and the error's std across the run is only ~1.0 m —
  i.e. it is almost a *pure constant offset*, not growing/accumulating
  error. Root cause identified below.
- Speed estimate noise visibly grows as the vehicle gets smaller/farther
  in-frame (see the widening oscillation in the speed plot after ~t=2s) —
  expected, but not yet quantified or folded into the confidence output.
- No real-world (or even non-toy-synthetic) footage has touched the
  pipeline yet — only one clean, idealized scene.

## Root cause of the position bias (important)

The dash-to-world-X assignment assumes the nearest dash visible in the
reference frame is dash index 0 of the pattern (i.e. that we're seeing the
start of the road markings). In this scene the road markings actually
start further back than the camera's visible frame + vehicle occlusion
allow us to see, so the pipeline anchored to the wrong dash index — a
constant ~2.3-period (27.4 m ÷ 12 m) misassignment. Because the *spacing*
between dashes is still correctly known regardless of which one is "dash
0," relative motion (speed) stays accurate while absolute position/
displacement inherits a fixed offset from the anchor error.

## Assumptions — valid vs dangerous

**Valid for this scene, and likely fine for many real ones:**
- Zero camera roll (image row depends only on world forward-distance).
- Straight road within the analysis window.
- Standard lane half-width (1.85 m) as declared, explicit prior.

**Dangerous if left as-is:**
- **Dash-index anchoring** (which visible dash is "index 0") — this is the
  cause of the position bias above and will misfire on any real footage
  where the road-marking start isn't visible, which is the common case.
  Needs either (a) an independent anchor (a stationary landmark of known
  position, a stop line, a visible intersection), or (b) reframing the
  output around **relative displacement from the first tracked frame**
  rather than absolute world position — which sidesteps the whole
  problem for speed/trajectory-shape purposes.
- **Standard lane width / dash pattern as hard-coded constants** — real
  roads vary by jurisdiction and road class. This must become a declared,
  uncertainty-weighted input (already flagged in code comments) rather
  than a silent constant, and the reconstructability score must penalize
  confidence when no independent scale check confirms it.
- **Color-blob vehicle tracking** is a synthetic-scene placeholder only;
  it has zero chance of working on real footage and must be replaced in
  Week 3 (a real detector — even a simple background-subtraction or
  lightweight learned detector — is P0 before touching real video).

## Current reconstruction error (summary)

| Metric | Value |
|---|---|
| Speed error (mean vs true) | 3.3% (19.33 vs 20.0 m/s) |
| Speed MAE / RMSE | 0.68 / 0.88 m/s |
| Position MAE / RMSE | 27.45 / 27.47 m (near-constant bias, see above) |
| Homography RANSAC inlier fraction | 69% |
| Vanishing point stability (bootstrap std) | ±4.2 px |
| Frames tracked / lost | 147 / 0 |

## Biggest technical risk

The dash-anchoring assumption is a **silent failure mode**: nothing in the
current pipeline detects that it has anchored to the wrong dash index —
the homography fit succeeds and looks confident (69% inliers) while
absolute position is wrong by a fixed ~27 m. A reconstructability score
that only looks at fit quality (as this one currently would) will not
catch this. The score needs an *independent* scale/anchor cross-check
(e.g. reconciling dash-based scale against a second cue such as known
vehicle dimensions, or explicitly reporting absolute position as
low-confidence whenever no independent anchor is available) before it can
be trusted the way section 16 requires.

## What must be solved before proceeding to Week 2

1. Fix or explicitly flag the dash-anchoring ambiguity — either add an
   anchor-independent mode (report displacement/speed only, mark absolute
   position as unavailable) or a second scale cross-check.
2. Quantify how tracking/estimation noise scales with vehicle size in
   frame (distance) and feed it into the uncertainty model — the growing
   oscillation in the speed plot is real signal for section 15's
   uncertainty propagation, not just noise to smooth away.
3. Replace the color-blob tracker with something that has a chance on
   real video before any real footage is attempted (Week 3 item, but
   worth prototyping early since it changes what "ground-contact point"
   noise looks like).
4. Build 2-3 more synthetic scenes with deliberately violated assumptions
   (curved road, camera roll, occluded lane markings) to stress-test the
   rejection path beyond the one blur case tested here.

## Verdict

**Feasibility confirmed.** The core geometry → scale → trajectory → speed
chain works on a controlled scene with no ground-truth leakage into the
pipeline (all camera parameters were re-derived from pixels, never read
from the generator), speed accuracy is good (3.3% error) with the
simplest reasonable smoothing, and the reconstructability gate correctly
refuses degraded input. The main real finding is that **absolute
position is currently untrustworthy** for a reason now understood and
fixable — exactly the kind of thing this feasibility phase exists to
surface before five more weeks are spent on top of it.
