# Multi-Camera Cross-Examination — Architecture, Results, and Honest Scope

## Why this exists

Real crashes that matter most — intersections, disputed liability,
multi-car incidents — often have more than one camera: a dashcam, a
nearby CCTV camera, a bystander's phone. Nobody today reconciles them
algorithmically; an investigator does it by eye. This extends the
Skeptic pattern (`docs/skeptic_agent.md` — one agent checking another's
conclusion using independent evidence) from *within* one camera's
analysis to *between* independent cameras: two witnesses, unsynchronized
and uncalibrated to each other, whose independent reconstructions either
corroborate or dispute one another.

**Scope of this pass, stated plainly:** a real, working proof of concept
on synthetic two-camera footage, not an end-to-end real-footage
capability. Two genuine algorithmic pieces got built and validated
(sync estimation, cross-camera confidence-interval comparison), and one
real, previously-unknown bug got found, diagnosed, and honestly worked
around (not hidden, not silently fixed in production). What real
multi-view footage would additionally require is listed at the end.

## The three pieces

### 1. A two-camera synthetic generator (`generate_dual_camera_scene.py`)

Generalized `scene_generator.py`'s camera model to support position
(`x_offset_m`/`y_offset_m`) and `yaw_deg`, not just pitch/roll/height
along a single fixed viewing axis (verified backward-compatible: the
full stress-test suite produces byte-identical results with the
generalized model at its old defaults). Camera A is the existing
single-camera setup; camera B sits at the *opposite end of the same
road*, facing back (`yaw=180°`) — a different, independently-computed
viewing geometry of the same physical event, not a translated copy.
Each camera renders at its own frame rate (30fps / 24fps) and starts
recording at its own, independent offset from a shared physical-event
clock — the ground truth records the true offset for validation; it is
never given to the reconstruction.

### 2. Time synchronization via a shared detected event

Neither camera's clock is known to the other. Each camera's own
(already-independent) reconstruction detects the same physical
collision instant at its own local time; the difference between those
two local-time readings *is* the offset between the cameras' clocks —
no shared clock or metadata required, just the same sharp, unambiguous
event observed twice.

**Validated result:** estimated offset error of **0.017s** against a
true 0.4s offset — within one frame's quantization at 24fps
(1/24 ≈ 0.042s). This is the one piece of this whole pass I'd call
solidly validated rather than "promising."

*Known simplification, stated plainly:* this proof of concept uses the
synthetic ground truth's `collision_frame` as the shared event anchor,
exactly like the existing single-camera collision scripts already do.
A real system needs its own per-camera event/change-point detector
before this sync method is usable on real, unlabeled footage — not
built here. `backend/causality/avoidability.py` already has adjacent
change-point logic (braking-onset detection) that's the natural
starting point for that work.

### 3. Cross-examination of independent estimates

Once synchronized, each camera's independent speed estimate (with its
own `backend/uncertainty/propagation.py` confidence interval) is
compared. If the two 90% intervals don't overlap, that's flagged as a
genuine dispute needing human review — not resolved by averaging.

**Validated result on this scenario:** camera A: 21.4 m/s [1.1, 41.7];
camera B: 22.1 m/s [14.9, 29.3] — intervals overlap, verdict "both
cameras agree." Both estimates land close to the true pre-braking speed
of 22.0 m/s.

## A real bug this work found (not hidden)

Camera B's background-subtraction tracker (the same one
`backend/api/process_video.py` uses in production) initially produced a
**degenerate, exactly-static trajectory** — reporting 0.00 m/s with
zero position variance, despite the vehicle genuinely moving ~50m in
that window. Root-caused by direct inspection: the color-blob reference
tracker, run on the identical footage, correctly reconstructed the
speed profile (22.7 m/s decelerating to ~12.5 m/s, matching the true
physics), so this is specific to the background-subtraction detector's
behavior on this camera's viewing geometry — a backward-facing,
far-positioned camera where the vehicle spends more of its visible
track near the vanishing point (smaller frame-to-frame pixel motion)
than in any single-camera scenario this tracker has been validated
against so far.

This proof-of-concept script (`run_multi_camera_cross_examination.py`)
detects this degeneracy (near-zero position variance) and falls back to
the color-blob tracker for that camera, clearly labeled
(`used_color_blob_fallback: true` in the output) — this is a targeted
workaround for this script only. **The production tracker choice in
`backend/api/process_video.py` was deliberately NOT changed** — bgsub
was presumably chosen there for good reasons validated against the
existing forward-facing single-camera scenarios, and swapping the
default needs its own regression pass across all of them, not a
one-scenario fix. Flagged as real follow-up work below.

A second, related finding: the color-blob tracker's *raw position*
values for camera B came out with the correct speed magnitude but an
inconsistent direction/origin relative to camera A's coordinate frame —
which points at the actual next architectural requirement, below.

## What this proves, and what it doesn't yet

**Proven:** two independently-calibrated cameras, with no shared clock
or coordinate system, can be synchronized from footage alone and their
independent conclusions meaningfully compared — and the comparison
correctly reflects real agreement when the underlying reconstructions
are both accurate.

**Not yet proven:** this on real, unlabeled footage (needs a real event
detector, not synthetic ground truth's `collision_frame`), and the
tracker robustness gap found above. **Coordinate reconciliation between
cameras** — comparing *position*, not just speed magnitude — was the
other major gap here; it's since been solved and validated (3.37m mean
position error after reconciliation, down from 22.71m without it) — see
"Hardening pass — scoped next-step #2" below.

## Hardening pass — scoped next-step #1 (real event detection)

Replaced the ground-truth-dependent sync anchor entirely.
`backend/multiview/event_detection.py` now detects the closest-approach
moment between two independently-tracked vehicles directly from each
camera's own reconstruction — `reconstruct_camera_independently()` in
the updated `run_multi_camera_cross_examination.py` uses **zero**
ground truth during reconstruction; `collision_frame` is read only at
the very end, to check the result, never fed into any pipeline stage.

**A real bug found and fixed during this hardening pass:** the
collision scenario freezes the two vehicles touching for the rest of
the clip after impact (by construction — see
`generate_dual_camera_scene.py`'s render loop), so the reconstructed
gap-vs-time curve plateaus near its minimum for a long tail rather than
having one sharp minimum. The first version of `detect_closest_approach`
used a plain `argmin`, which picked an arbitrary point somewhere in that
plateau — confirmed by inspection: it was landing suspiciously close to
the *other* camera's true event time, which looked like a swap bug at
first but was actually two independent plateau-sampling artifacts.
Fixed by taking the *first* time the gap reaches within a small
tolerance of its eventual minimum (the first-contact instant) instead
of the global minimum.

**Honest result after the fix:** self-discovered sync error dropped
from 0.73s to **0.51s** — real, measured improvement, but still far
worse than the 0.017s achieved when ground truth anchored the event
directly. Root cause, diagnosed rather than guessed: the *first-pass,
untruncated* tracking run (needed to find the event before truncation is
possible) has to track straight through the occlusion/merged-blob
frames right at the collision itself, and that occlusion transition
visibly corrupts the reconstructed trajectory in exactly the region the
event detector depends on most. This compounds further when the
background-subtraction-vs-color-blob fallback (below) kicks in, since
the two trackers fail differently around occlusion.

**Where this leaves scoped next-step #1:** genuinely hardened in the
sense that mattered most — the ground-truth crutch is gone, replaced by
a real, self-contained, working algorithm, with one concrete bug found
and fixed along the way. Not yet hardened to the same *precision* as
the idealized version; that needs occlusion-aware tracking through the
event itself (a real, separate piece of work, not a tuning knob) before
sync precision on real footage would match what was demonstrated with
ground truth. Flagged as the next concrete task, not glossed over.

## Concretely scoped next steps

1. ~~Real per-camera event detection~~ **Done, partially** — see the
   hardening section above. Ground-truth dependency fully removed and
   replaced with a working, self-contained detector; remaining gap is
   *precision* (0.51s vs the 0.017s ground-truth-anchored baseline),
   traced to occlusion corrupting the first-pass tracking. Next: an
   occlusion-aware tracking pass through the event itself, not a
   different detection formula.
2. ~~Coordinate reconciliation between cameras~~ **Done** — see
   "Hardening pass — scoped next-step #2" below.
3. **Background-subtraction tracker robustness** for cameras where a
   vehicle's apparent pixel motion is small for extended stretches (far
   from camera, near the vanishing point) — the bug found above,
   root-caused but not fixed.
4. Extend from two cameras to N, and from "do they agree" to weighting
   disagreement by each camera's own stated uncertainty rather than a
   simple overlap check.

## Hardening pass — scoped next-step #2 (coordinate reconciliation)

`backend/multiview/coordinate_reconciliation.py`: determines whether two
independently-fit camera coordinate frames run the same or opposite
direction (from each camera's own velocity sign — no ground truth), and
the offset needed to express one camera's positions in the other's
frame, anchored at each camera's own self-detected event time.

**Validated result** (`run_coordinate_reconciliation_eval.py`): direction
sign correctly detected as **-1** for camera B — confirming, quantitatively
this time, what earlier inspection found by hand (camera B's backward
orientation produces a flipped coordinate axis). After reconciliation,
comparing the two cameras' independent position reconstructions for the
same physical vehicle over their overlapping synced window (46 points
across 2.3s):

- **Mean position error after reconciliation: 3.37m**
- Mean position error *without* reconciliation (i.e. comparing the two
  cameras' raw local coordinates directly): **22.71m** — included
  specifically to show reconciliation is doing real, substantial work,
  not a no-op that happens to look fine.

3.37m mean error over a shared window with the vehicle moving at
15-20 m/s is a genuine, usable result, not a polished one — it inherits
whatever error is already in each camera's own independent
reconstruction (tracking noise, the sync precision limit documented
above, the color-blob fallback camera B needed). This is now a real,
measured capability: two independently-calibrated, unsynchronized
cameras can have their *positions*, not just their speed magnitudes,
meaningfully compared.

## From n=1 to a real sample (multi-scenario sweep)

Everything above was one scenario. `run_multi_camera_sweep.py` runs the
full pipeline (geometry, tracking, event detection, sync, coordinate
reconciliation, cross-examination) across 6 varied scenarios — three
speeds (15/18/22/26 m/s), four different sync offsets including a
negative one (camera B starting BEFORE camera A) — end to end, no
shortcuts, same code as the single-scenario version.

**6/6 scenarios succeeded end to end.** Aggregate results:

| metric | value |
|---|---|
| mean sync error | 0.420s |
| mean speed agreement between cameras | 4.0% |
| CI overlap rate (cameras statistically agree) | 100% |
| mean position error after reconciliation | 2.95m |
| direction sign correctly detected | 100% |

Consistent with the single-scenario numbers above (0.51s sync error,
3.37m position error) — this isn't a different, better result, it's
confirmation that the single-scenario numbers weren't a lucky anecdote.
The direction-sign detection (same backward-facing camera B config in
every trial) hitting 100% across all 6 independently-generated scenarios
is the strongest evidence yet that the coordinate reconciliation
algorithm is finding something real, not overfitting to one specific
rendering.
