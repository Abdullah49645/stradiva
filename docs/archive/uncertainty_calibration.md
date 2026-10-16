# Uncertainty Calibration — Confidence Interval Coverage

**Scope:** turn the pipeline's existing diagnostics into a stated speed
confidence interval (`backend/uncertainty/propagation.py`), then measure
whether that interval is actually trustworthy using the brief's own
metric (section 25): **confidence interval coverage** — how often the
true speed falls inside the stated 90% interval.

## Method

`evaluation/scripts/run_uncertainty_calibration.py` generates 28
synthetic single-vehicle trials (4 speeds × 7 roll angles, 0°/±3°/±6°/±9°),
runs each through the real pipeline (agentic geometry search, roll
correction, tracking, speed reconstruction), computes the stated 90% CI
via `propagate_speed_uncertainty`, and checks containment against the
scene's known ground-truth speed.

**A reproducibility bug was found and fixed while building this**:
`scene_generator.py`'s per-name deterministic scene seeding used
Python's built-in `hash(name)`, which is salted with a random per-process
seed by default (PYTHONHASHSEED, a CPython security feature unrelated to
this use case) — so "same scenario name → same scene" was silently false
across separate process runs, even though it held within a single run.
This affected every named-scenario script in the repo (stress tests,
this calibration sweep, demo-data generation), not just this one. Fixed
by switching to `zlib.crc32(name.encode())`, which has no per-process
salt. The numbers below are from the run *after* that fix; an earlier,
pre-fix run of this same sweep produced different per-trial results
purely due to unintentional reseeding, which is why you may see
different numbers if you find an earlier copy of this doc.

## Results

- **19/28 trials reconstructed** (9 rejected by the reconstructability
  gate — up from 17/28 before the neighbor-frame search agent upgrade;
  see `docs/agent_tool_upgrades.md` for that change and an important
  reverted regression found while trying to push this further).
- Coverage at the pre-calibration default (`CI90_SCALE=1.0`): **68.4%**
  — the stated interval was too narrow, i.e. overconfident.
- Swept `CI90_SCALE` over {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0}:

  | scale | coverage |
  |---|---|
  | 0.5 | 63.2% |
  | 0.75 | 63.2% |
  | 1.0 | 68.4% |
  | 1.25 | 73.7% |
  | 1.5 | 78.9% |
  | 2.0 | 84.2% |
  | **2.5** | **94.7%** |
  | 3.0 | 94.7% |

- `CI90_SCALE = 2.5` is now the default in `propagation.py` — closest
  grid point to the 90% target. 94.7% vs. 90% on 19 trials is one
  trial's difference either way — a reasonable first calibration, not a
  tight one. Refit as the trial set grows.

## Open finding from this sweep (not previously known, still open)

9 of 28 trials — mostly small roll angles (±3° to ±6°) — still come back
`NOT_RECONSTRUCTABLE`. Root-caused in `docs/agent_tool_upgrades.md`:
mild roll degrades RANSAC inlier fraction diffusely across every
candidate frame without tripping the width-consistency check that the
roll-correction hypothesis currently relies on to trigger. A fix was
attempted and reverted after it regressed the canonical validated
scenario — see that doc for what a real fix would need (an independent
cross-check on the accepted correction angle, not just a looser
trigger). Raw per-trial data is in
`docs/uncertainty_calibration_results.json`.
