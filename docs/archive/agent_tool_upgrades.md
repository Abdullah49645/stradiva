# Agent Tool Upgrades — Roll-Search Robustness

**Motivation:** the uncertainty calibration sweep (`docs/uncertainty_calibration.md`)
measured that the agentic roll search, which reconstructs the hand-built
8° roll stress scenario essentially perfectly, only succeeded on 17/28
trials (60.7%) across a randomized speed/roll grid. This doc covers what
was tried to close that gap, what shipped, and — just as importantly —
what was tried and reverted because it broke correctness on the
validated scenario.

## What shipped: neighbor-frame search

`backend/agent/loop.py`'s original candidate search only tries 8 fixed
frame positions (0.75, 0.5, 0.25, 0.125, ... of the clip). Diagnosing
individual failing trials showed the dominant rejection reason often
wasn't camera roll at all — it was simply landing on one of those 8
fixed frames at a point in the road-dash cycle with too few visible
markings (as few as 6 correspondence points spanning 3m, against a
20-point / 30m requirement), while a few frames away the same stretch of
road would have had a clearer view.

Added: after the fixed grid is exhausted, search frames immediately
around whichever candidate scored best so far (offsets of
2/4/6/8/12/16/20/28 frames, capped at 8 additional tries) — a genuine
"go get more evidence near where the evidence already looked most
promising" step, not a wider blind grid. This is a real PERCEIVE →
ASSESS → IDENTIFY MISSING EVIDENCE → SELECT NEXT VISUAL ANALYSIS → RUN
TOOL loop, distinct from the existing roll-angle search.

**Result:** 17/28 → **19/28** reconstructable (60.7% → 67.9%) on the
same calibration grid, with the canonical roll stress scenario
unchanged (still 0.03% speed error) — confirmed by rerunning
`evaluation/scripts/run_stress_tests.py` before and after.

## What was tried and reverted: broadening the roll-suspect trigger

The remaining failures were diagnosed further: several small-roll trials
(±3° to ±6°) fail not on correspondence sparsity but on RANSAC inlier
fraction dropping below threshold *diffusely, across every candidate
frame* — a signature the original roll-suspect trigger (a large
lane-width self-consistency error on one specific frame) doesn't catch,
because mild roll can degrade inlier fraction without ever tripping the
width-consistency check.

The obvious fix — also flag adequate-correspondence-but-low-inlier
frames as roll suspects, not just high-width-error ones — was
implemented and **immediately caused a serious regression**: the
canonical 8° roll stress scenario, previously reconstructing at 0.03%
speed error, jumped to **91% speed error**. The broadened trigger let
the roll-angle search accept a wrong angle that happened to coincidentally
pass both the width-consistency and correspondence-span checks on a
different, incorrect frame — a false positive the original, more
conservative trigger was specifically preventing.

This was reverted in the same work session, verified by rerunning the
full stress suite (back to 0.03% error), and is documented here rather
than silently dropped so a future session doesn't reintroduce it without
first fixing the actual weakness it exposed: **the width-consistency +
span check, while good, is not sufficient on its own to rule out a
wrong-angle false positive** — a genuine, still-open problem that a
looser trigger only makes worse by evaluating more candidate angles
against the same insufficient acceptance test. A real fix would need a
second, independent validation signal, e.g. checking that the corrected
homography's predicted vehicle width or lane curvature also stays
physically plausible at the far end of the correspondence span, not just
within it. Not attempted here — flagging as the next concrete piece of
work on this system rather than shipping a guess.

## Net effect

- Reconstructable rate on the calibration grid: 60.7% → 67.9% (real,
  measured, no regression).
- `CI90_SCALE` refit accordingly (2.5, coverage 94.7% on 19 trials — see
  `docs/uncertainty_calibration.md`).
- One concrete, well-scoped next task identified for whoever picks this
  up next: an independent cross-check for the roll-correction search's
  accepted angle, not just a looser acceptance threshold.
