# Pre-registration: v2.2 pipeline, investigator and near-miss measures

Committed before any of the runs below. The git commit adding this file is the timestamp. Changes
after it are listed under "Deviations".

## What changed since v2.1 (developed on dev only)
- **Investigator** (`backend/investigator/investigator.py`): when the Skeptic blocks a result, it runs
  the remedy for the failure named (camera bump → analyse the stable parts; marking-proportion
  mismatch → every documented standard with matching proportions, verdict only if it holds under all;
  impossible motion of a subject vehicle → quarantine those measurements and re-run). A remedy is
  adopted only if every blocking check then passes and a same-lane pair is reconstructed.
- **Near-miss measures** (`analyze.py::_near_miss`): minimum bumper gap, minimum time to collision
  (capped at 10 s), maximum deceleration rate to avoid the crash, each with a 90 % interval, over
  the interval in which both vehicles were measured (pre-contact for collisions).
- **v2.2 pipeline fixes**: a "pushed lead" impact signature is used only while the lead stays in
  view (traffic approaching, or receding with the lead never hidden); the resolution gate is measured
  at the event (contact, else closest approach) instead of at the follower's last position.

Development used: dev split (seeds 1000+), oracle_clean dev for the near-miss measures, and the
Skeptic's fault-injection scenes 0-7 (seeds 700-707) for the investigator.

## Held-out runs (each run once, on the frozen code, reported whatever the result)
1. **test3**: new split, seeds 17000+i, i = 0..29, same distribution as test/test2. All three layers
   (oracle_clean, oracle_occluded, e2e). e2e runs with `--investigate`, recording the base
   pipeline's outcome and the investigator's outcome on the same clip.
2. **Fault injection, held-out scenes**: `investigator_faults.py --start 8 --n 8` (seeds 708-715),
   all seven faults (including the negative controls).

## Metrics
- Outcome categories and the harmful set exactly as in `summarize_suite.py` (unchanged).
- Investigator: number of clips whose outcome it changed; of those, changed to correct, to harmful,
  to another abstention. Harmful cases created by the investigator are reported individually.
- Near-miss: on clips where CARAI reports the measure and ground truth defines it over the same
  interval, median absolute error and 90 % interval coverage of min gap, min TTC (both capped at
  10 s) and max DRAC, per layer.
- Faults: per fault, answered by base vs final, answered and correct (worst speed error ≤ 10 % or
  every interval covers the truth), harmful (answered, unflagged, worst error > 10 %).

## Deviations
(none yet)
