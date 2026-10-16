# Week 2 Assessment — Geometry, Calibration & Reconstructability

**Scope:** fix the Week 1 finding (absolute position bias), build the
reconstructability scoring subsystem (section 16), and stress-test the
whole gate against a small deliberate-failure dataset (section 26) instead
of just the one hand-picked scene from Week 1.

## 1. Fixed: the dash-anchoring position bias

Week 1 found absolute position was off by a near-constant ~27 m because
the pipeline assumed the nearest visible dash was pattern-index 0. Fix:
**report relative displacement from the first observed frame as the
primary, higher-confidence output**, and explicitly label absolute
position as unanchored/low-confidence unless an independent landmark is
supplied (none is, yet — that's a real, open limitation, not fixed here).

Result on the baseline scene:

| Metric | Before (absolute) | After (relative displacement) |
|---|---|---|
| MAE | 27.45 m | **0.87 m** |
| RMSE | 27.47 m | **1.31 m** |

Speed accuracy was unaffected either way (it was never the problem) —
3.3% error, confirming speed is robust to the anchor question while
absolute position genuinely isn't.

## 2. Built: the reconstructability scorer (`backend/reconstructability/score.py`)

Four weighted components (geometry, scale, tracking, temporal) plus hard
floors that veto regardless of the blended average, plus a lane-width
self-consistency check (reproject the fitted edges through the recovered
homography at multiple depths; a correct homography should recover ~3.7 m
consistently at every depth — drift or offset there means the ground-plane
mapping itself is untrustworthy, independent of the anchor question).

## 3. Stress-tested against 5 scenarios — and found a second real bug

Built `evaluation/scripts/generate_stress_scenes.py` /
`run_stress_tests.py`: same camera/road/vehicle model as scene A, with
one variable changed per scenario (brief section 26's "deliberate-failure
dataset" concept). Actual results:

| Scenario | Expected | First-pass verdict | Verdict after width-consistency veto | Verdict after active de-rotation (current) | Notes |
|---|---|---|---|---|---|
| A — baseline | RECONSTRUCTABLE | ✅ RECONSTRUCTABLE (87.2) | — | — | speed err 2.2% |
| B — no scale reference (edges only, no dashes) | NOT_RECONSTRUCTABLE | ✅ NOT_RECONSTRUCTABLE (34.5) | — | — | correctly refuses, no correspondence points |
| C — 8° camera roll | should be flagged/rejected, then reconstructed if corrected | ❌ **RECONSTRUCTABLE (83.0)** despite **45.3% speed error** | ✅ NOT_RECONSTRUCTABLE (vetoed) | ✅ **RECONSTRUCTABLE (79.6)**, **0.03% speed error** | see below — active correction, not just rejection |
| D — heavy fog | NOT_RECONSTRUCTABLE | ✅ NOT_RECONSTRUCTABLE (15.0) | — | — | no usable line segments at all |
| E — faster vehicle (32 m/s), otherwise identical to A | RECONSTRUCTABLE | ✅ RECONSTRUCTABLE (87.2) | — | — | speed err 1.9% — confirms A wasn't a lucky single-speed fit |

(Current numbers as of the latest `evaluation/scripts/run_stress_tests.py` run, saved to
`docs/week2_stress_test_results.json`. The "first-pass" and "after veto" columns are the
original Week 2 findings described below; they're kept for the narrative even though the
absolute score values were later re-baselined slightly by unrelated scoring changes.)

### The camera-roll finding is the real result of this week

Scenario C violates the zero-roll assumption baked into every geometry
module so far (see the docstring in `scene_generator.py`/
`generate_synthetic_scene.py`: "image row depends only on world
forward-distance for a zero-roll camera" — this stops being exactly true
under roll). The homography still fit with 67% RANSAC inliers and decent
VP stability, so geometry/tracking/temporal all scored well — but the
recovered speed was **45% wrong**, and the lane-width self-consistency
check caught it clearly (2.08 m error vs 3.7 m assumed width, vs 0.66 m
on the clean baseline).

The bug wasn't in detecting the problem — it was in how the *score*
used that detection. A weighted average let three healthy components
(geometry 98, tracking 100, temporal 100) dilute one bad one (scale
45.2) down to a passing 83/100. **A confidently-wrong reconstruction
would have shipped.** This is precisely the failure mode section 16
and the "never fake certainty" principle warn about, and it wasn't
hypothetical — it happened on the first stress-test run.

**Fix applied:** the lane-width self-consistency error now vetoes
reconstruction independently of the blended score whenever it exceeds
1.0 m, regardless of how well other components scored. Re-running
scenario C after the fix: **NOT_RECONSTRUCTABLE**, correctly.

This generalizes to a design rule worth carrying into every future
component: **a strong direct diagnostic for a specific failure mode
should be able to veto on its own — it must never be only a minority
vote in a weighted average that healthier components can outvote.**

## 4. Update: roll is now actively corrected, not just detected/rejected

The veto above (item 3 in the original "what's still open" list, quoted
in this section's git history) was the honest interim fix: it stopped a
confidently-wrong reconstruction from shipping, but it meant every
rolled video — a common case for handheld/dashcam sources — was simply
rejected.

`cv/geometry/roll_correction.py` replaces that with active de-rotation:
a coarse-to-fine search over candidate roll angles, reusing the *same*
lane-width self-consistency check from item 3 as the search's objective
function (the angle that minimizes width-reconstruction error is taken
as the correction) rather than deriving the correction's sign/convention
analytically. This is integrated into `backend/agent/loop.py`'s unified
agentic search, not a one-off special case, and both `evaluation/scripts/
run_feasibility.py` and the production entry point (`backend/api/
process_video.py`) derotate the full frame sequence by the winning angle
before tracking, since the fitted homography is only valid in the
derotated coordinate space.

While building this, a second issue surfaced: width-consistency alone
can accept a homography calibrated over a suspiciously narrow depth
range (e.g. correspondences clustered around 0–15 m) that then
extrapolates badly to the actual vehicle region further down the road.
This is now caught by a minimum correspondence-span check
(`MIN_CORRESPONDENCE_SPAN_M` in `backend/agent/loop.py`, `world_x_span_m`
in `backend/agent/tools.py`) applied alongside the width-consistency veto.

**Result on scenario C (8° camera roll):** speed error went from
**45.3% → 0.03%**, and the scenario now correctly resolves to
RECONSTRUCTABLE (an essentially-perfect reconstruction) instead of being
rejected outright. The other four stress scenarios (A, B, D, E) showed
no regressions — see the updated table above and
`docs/week2_stress_test_results.json` for the current run's raw numbers.

## What's still open going into Week 3

1. ~~Roll is detected, not corrected.~~ **Resolved** — see section 4
   above. Remaining related work: the correspondence-span threshold
   (30 m) and the angle-search range/step are reasonable first guesses,
   not calibrated against a larger sweep of roll angles. Partially
   addressed — see item 5 below.
2. **Absolute position still has no independent anchor.** Relative
   displacement is solid; a real forensic conclusion (e.g., "vehicle
   traveled from point A to point B") eventually needs one. Deferred —
   this needs either scene-specific landmarks or accepting relative-only
   output as the honest deliverable for many videos.
3. **The veto threshold (1.0 m) is a reasonable first guess, not
   calibrated.** Needs a larger scenario sweep (varying roll angle,
   pitch error, focal-length error) to find where the self-consistency
   signal actually starts predicting real speed error, rather than one
   anecdote at 8°.
4. Vehicle tracking is still the Week-1 color-blob placeholder — Week 3's
   real detector/tracker work is unaffected by anything done this week
   and remains next.
5. ~~The width-consistency + span check isn't a sufficient validity test
   for the roll search on its own.~~ **Resolved** — see
   `docs/skeptic_agent.md`. Built an independent second-agent check (the
   Skeptic) specifically to catch this; verified it against the exact
   reverted false positive (0.03% → 91% error) by temporarily
   reintroducing that bug and confirming the Skeptic flags it
   (`skeptic_disputed: True`, depth-trend R²=1.00), then confirming it
   does NOT flag the correct, validated 0.03% result on the same
   scenario. Also surfaced a genuine, smaller previously-unknown finding
   in the process — see that doc.

## Verdict

Geometry/calibration/scale recovery holds up under a small deliberate-
failure sweep, and — more importantly — the sweep did its job: it found
a real, non-obvious scoring bug (a violated assumption hiding behind a
healthy weighted average) before any of this reached real footage. That
bug is now fixed and covered by a regression scenario
(`synthetic_scene_C_camera_roll`) that stays in the stress-test suite
going forward. That scenario has since moved from "correctly rejected"
to "correctly and accurately reconstructed" (section 4), which is the
stronger outcome the veto alone couldn't deliver.
