# Week 3–4 Assessment — Collision Causality & Avoidability Analysis

**Scope:** implement the new required capability (project owner's
addendum): given two reconstructed vehicle trajectories, determine
pre-impact kinematics and whether the collision appears physically
avoidable, with the same "never fake certainty" discipline as the rest
of the pipeline. Per the addendum, week numbers are ordering guidance,
not a hard schedule — this work sits alongside remaining Week 3 items
(a real vehicle detector is still outstanding, see below) rather than
strictly after them.

Full run: `evaluation/scripts/run_causality_feasibility.py`, using two
new controlled two-vehicle scenarios (`generate_collision_scene.py`)
with independently-computed ground-truth avoidability answers.

## What was built

- **Two-vehicle collision scenario generator** — reuses the Week 1/2
  camera/road model, adds a lead vehicle (stopped in-lane, green) and a
  following vehicle (blue... rendered red for tracker reuse) that
  brakes at a configurable time/strength. Each scenario also computes
  an **independent, physics-only ground-truth avoidability answer**
  (`required_stopping_distance = v·t_reaction + v²/(2·a_max)` vs. the
  actual gap at the earliest hazard-visible point) so the module's
  output can be checked against something not derived from the CV
  pipeline at all.
- **Multi-vehicle tracking** — extended the Week 1 color-blob tracker
  to a second HSV range (still a synthetic-scene placeholder, same
  caveat as Week 1: this must be replaced before real footage).
- **`backend/causality/avoidability.py`** — the actual new capability:
  braking-onset detection (sustained deceleration threshold), a
  backward scan for the last instant at which the required stopping
  distance still fit the observed gap ("last possible avoidance
  point"), and an `AVOIDABLE` / `NOT_AVOIDABLE` / `INSUFFICIENT_EVIDENCE`
  verdict with a human-readable narrative in the format the addendum
  specified. Physical assumptions (max deceleration, reaction time) are
  explicit fields on every result, never hidden constants. The module
  never outputs fault language — only physical/evidence-based claims.
- **Uncertainty inheritance** — the verdict is gated by the upstream
  reconstructability confidence (mapped from the section-16 score); a
  synthetic LOW-confidence input correctly forces
  `INSUFFICIENT_EVIDENCE` regardless of what the kinematics say.
- **Docs updated per the addendum's explicit instruction**:
  `docs/definition_of_done.md` and `docs/priorities.md` rewritten to
  insert this capability into P0 and into the "what ships" definition,
  not bolted on as an appendix.

## Validation against ground truth

| Scenario | Ground-truth physics answer | CARAI verdict | Agree? |
|---|---|---|---|
| `unavoidable` (25 m/s, 50.5 m gap, weak/late braking) | required 74.6 m > available 50.5 m → **not avoidable** | **NOT_AVOIDABLE** — "required stopping distance... exceeded the available gap... even under immediate maximum braking" | ✅ |
| `avoidable_but_occurred` (18 m/s, 55.5 m gap, braking started 2.3s before impact but too weak) | required 44.7 m < available 55.5 m → **avoidable in principle** | **AVOIDABLE** — "~2.5s of potential avoidance time... braking began approximately 2.3s before impact" | ✅ |

Both scenarios reconstructed with reconstructability scores in the
high-80s/low-90s, and (importantly) the module correctly distinguished
"physically impossible to avoid" from "physically possible but not
actually avoided" — the two questions the addendum's example output
depends on being kept separate.

## Two real bugs found and fixed during this pass (not projected — actually happened)

1. **Post-collision frame contamination.** Initially tracked vehicles
   through the collision frame and a few frames after (needed for
   visual confirmation of impact). The post-collision "frozen" position
   fed into the Savitzky-Golay smoother produced a fake deceleration
   artifact right before impact — exactly where the avoidability
   assessment needs clean data most. Fixed by truncating tracking input
   to strictly pre-collision frames.
2. **Near-camera homography extrapolation error.** The very first ~15%
   of a vehicle's tracked samples (when it's large/close, near the
   bottom of the frame) showed a spurious speed ramp-in (12→21 m/s
   climbing toward a true constant 25 m/s) rather than the correct flat
   line. Root cause: the ground-plane homography's correspondence
   points (from the dashed centerline) don't extend to the very bottom
   rows of the image, so close-range positions are extrapolated beyond
   the calibrated region rather than interpolated within it. Fixed with
   an explicit, reported discard of the earliest ~15% of pre-impact
   samples — a real limitation, not silently patched: it's logged in
   the causality result's evidence list every time it fires.

Both bugs were caught only because two full scenarios were run and
checked against independent ground truth, not because they were
anticipated — consistent with the project's own stress-testing
discipline from Week 2.

## What's still open

1. **Real vehicle detector/tracker** — still the Week 1 placeholder,
   now used for two vehicles instead of one. This is unaffected by the
   causality work and remains the next P0 item before any real footage.
2. **Near-camera homography accuracy** — the 15%-discard is a
   workaround, not a fix. A proper fix means extending scale
   correspondences closer to the image bottom (more/better near-field
   dash detections) or modeling the homography's reliability as a
   function of image row and feeding that into per-sample confidence
   rather than a blanket discard fraction.
3. **Hazard-onset detection.** The ground-truth physics check currently
   evaluates avoidability from t=0 of the clip (valid here because the
   lead vehicle is visible/stopped for the whole synthetic clip). Real
   footage needs an actual "when did the hazard become observable"
   detector — this is currently a stated simplification, not solved.
4. **Steering/evasive-maneuver detection** is not implemented yet —
   only braking is detected. The addendum asks for "braking, steering,
   or another evasive maneuver"; steering needs lateral (Y) trajectory
   analysis, which the module doesn't yet look at.
5. Report/UI/demo restructuring (per `docs/definition_of_done.md`) is
   specified but not yet built — no report generator or UI exists yet
   at all; this is queued alongside the rest of P1.

## Verdict

The core causality/avoidability claim in the addendum — that CARAI
should distinguish "physically avoidable" from "physically unavoidable"
using reconstructed motion, with proper uncertainty gating and never a
fault claim — is implemented and validated against independent physics
ground truth on two scenarios that were deliberately built to disagree
with each other. The module is real, not a stub: it found and survived
two genuine bugs in the surrounding pipeline during this pass.
