# The Skeptic — an Adversarial Second Agent

## The problem this exists to solve

Every check in this pipeline up to this point was written by, and graded
by, the same system that produces the answer. `backend/agent/loop.py`
(the Reconstructor) searches for a frame/angle whose geometry clears an
acceptance threshold, and the *same* threshold decides whether to trust
it. That's a structural blind spot: if the threshold has a gap, nothing
in the system can ever notice, because nothing checks the check.

This isn't hypothetical. While working through `docs/agent_tool_upgrades.md`,
broadening the Reconstructor's roll-suspect trigger caused it to accept a
**wrong** rotation angle that happened to coincidentally satisfy the
width-consistency and correspondence-span acceptance tests — 0.03% speed
error became 91%, silently, under a confident `RECONSTRUCTABLE` verdict.
It was only caught because a human was running an external regression
suite. The fix at the time was to revert and stay conservative. That's a
patch, not a solution — the same class of failure was still possible
anywhere the acceptance test had a gap the Reconstructor didn't know to
look for.

## The design

`backend/agent/skeptic.py` adds a second agent role — the **Skeptic** —
that runs after the Reconstructor accepts an answer and tries to find a
reason to disagree with it, using evidence the Reconstructor's own
acceptance test did **not** use:

1. **Angle ambiguity** (when roll correction was used): a real correction
   should have one clearly-best angle. If a candidate several degrees
   away also scores near the acceptance threshold, the "winner" isn't
   actually unique — the exact shape of the false positive above (a
   loosened test let a coincidentally-passing wrong angle through a
   field of close competitors).
2. **Depth-trend check**: the Reconstructor's width-consistency check
   reduces many sampled rows to one max-error scalar, which can hide a
   systematic bias if no single row crosses the threshold even though
   the errors trend consistently with depth (which is exactly what a
   residual, uncorrected rotation error looks like — physically, not
   randomly). This fits a line through per-row width error vs. depth and
   flags a real slope.

A dispute doesn't silently reject the result — the Reconstructor's
evidence was real. It downgrades confidence (`skeptic_disputed: true`,
with the specific finding attached) and surfaces the disagreement
directly: in `full_diag["skeptic_findings"]`, in the agent trace log, in
the rendered report's new "Cross-Examination" section, and in the
frontend's reconstructability panel. A human reviewer sees the argument,
not a hidden internal flag.

## Proof it actually works — reproducing the exact known bug

The strongest validation available: temporarily re-applied the exact
reverted broadened-trigger change from `docs/agent_tool_upgrades.md` and
re-ran the canonical 8° roll stress scenario with the Skeptic wired in.

**Before (no Skeptic, the bug from that doc):** `RECONSTRUCTABLE`, 91.1%
speed error, shipped with full confidence and no signal anything was
wrong.

**With the Skeptic added, same reintroduced bug:**
```
status: RECONSTRUCTABLE   speed_error: 91.1%   roll_correction_deg: -9
skeptic_disputed: True
 - angle_ambiguity: False (winning angle had no close rival)
 - depth_trend: True — "lane-width error trends -1.05m per 100m of
   depth (R^2=1.00) -- a systematic drift consistent with residual,
   uncorrected geometric bias, not random noise."
```
The depth-trend check — not the angle-ambiguity check — caught it, with
a near-perfect R²=1.00 signal. This is the check working exactly as
designed: an independent statistic the Reconstructor's own test never
computes, catching the specific failure mode that fooled the original
test.

**Then reverted the bug back out** (restored the correct, conservative
trigger) and reran the same scenario: `RECONSTRUCTABLE`, 0.034% speed
error (matching the previously-validated 0.03%), `skeptic_disputed:
False`, both checks cleanly passing. The Skeptic distinguishes the real
answer from the wrong one on the same scenario, using the same
independent signals — it isn't just a stricter copy of the original
test, and it doesn't cry wolf on the answer that's actually correct.

## A genuine new finding, surfaced honestly

Running the Skeptic across the existing demo scenarios (not a
manufactured test) turned up something real and previously unknown: the
`avoidable_but_occurred` collision demo — no camera roll involved at
all — also shows a systematic depth-trend bias (-0.76m per 100m depth,
R²=1.00). Checked against ground truth: true speed 18.0 m/s, estimated
17.36 m/s (3.6% error, comfortably inside the stated 90% CI of
8.1–26.6 m/s) — so the practical impact here is modest, not another
91%-style miss. But the bias is real and R²=1.00 means it's not noise;
it's a small, consistent geometric artifact in the dash-detection/
homography-fitting method itself, independent of roll, that the
existing single-scalar width-consistency check was never sensitive
enough to catch. Not chased down further in this pass — flagged
honestly here as a genuine, well-scoped next investigation rather than
either hidden or over-claimed as urgent.

## Why this is the right kind of upgrade, not just another feature

Everything else in this pipeline is CARAI stating how confident it is
about itself. The Skeptic is CARAI *checking* itself — a structurally
different and stronger claim, and one with a real, non-staged proof
behind it: an actual near-miss from this project's own development
history, caught after the fact by a second, adversarial process. That's
also a materially stronger Agentic Vision Award story than a single
search-until-threshold loop: two roles that can genuinely disagree,
where the second one's whole purpose is orchestrated skepticism of the
first, backed by a documented case where that disagreement was the only
thing standing between a shipped confident answer and a real,
demonstrated 91% error.
