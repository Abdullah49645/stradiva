# Demo Video Shooting Script

Follows the brief's own recommended 5-minute structure (problem → reconstruction →
physical result → agentic moment → failure case → evaluation → AWS → close). Every
beat below is mapped to a real, already-existing asset or an exact command to run live
— nothing here needs to be staged or invented. Where a beat needs a screen recording,
the exact file or URL to have open is given.

## 0:00–0:20 — Problem

Voiceover over `evaluation/dataset/evidence_unavoidable.mp4` playing (plain, no
annotations yet — or use the source video before annotation if you want it cleaner).

> "This is dashcam footage of a collision. You can see what happened — but not how
> fast anyone was going, or whether it could have been avoided. That footage was
> never calibrated for measurement."

## 0:20–1:10 — The reconstruction

Screen recording: open the web app (`npm run dev`, localhost:3000), on the
**Collision Reconstruction** tab, select **Case 001 — Rear-End, High Closing Speed**.
Let the page load — the hero verdict, the reconstruction quality panel, and the
Skeptic's cross-examination banner all render live.

> "CARAI doesn't assume the footage is measurable. It searches the frame for lane
> markings, recovers the camera geometry, and only then reconstructs motion."

Cut to `evaluation/dataset/evidence_unavoidable.mp4` playing — point out the live
telemetry chips (speed + trend arrows), the sidebar's 90% confidence interval, and
the SKEPTIC status line.

## 1:10–1:50 — Physical result

On-screen text, pulled directly from the actual pipeline output (no invented numbers):

```
Pre-impact speed: 24.4 m/s (88 km/h)
90% confidence interval: 13.8–35.1 m/s
Time to collision at earliest observation: 1.4s
Verdict: Appears Unavoidable
```

> "Not a bare number — a calibrated confidence interval. `docs/uncertainty_calibration.md`
> measures that this interval is actually right about 94.7% of the time it should be,
> across 19 independent trials — not just asserted."

## 1:50–2:30 — The agentic moment (the Skeptic)

This is the strongest, most authentic beat — it's a real bug this project's own
development history produced, not a staged demo. Screen-record this exact sequence:

```bash
cd carai
python3 evaluation/scripts/run_stress_tests.py
```
Point out scenario C: **0.034% speed error** on an 8° camera-roll scenario.

> "While building the roll-correction search further, we tried a change that looked
> reasonable — and it silently broke this exact scenario."

Cut to the numbers in `docs/agent_tool_upgrades.md`: the same scenario, same code
otherwise, **91% speed error** — accepted with full confidence, no warning.

> "We caught it by accident, from outside the system. So we built a second agent —
> the Skeptic — whose only job is to try to prove the first agent wrong, using
> evidence it didn't use to reach its answer."

Show the actual reproduction: re-apply the described change, rerun, and this time the
output includes:
```
skeptic_disputed: True
- depth_trend: "lane-width error trends -1.05m per 100m of depth (R^2=1.00) --
  a systematic drift consistent with residual, uncorrected geometric bias"
```

> "Same bug, same code — but now the system catches itself before it ships a wrong
> answer with full confidence."

## 2:30–3:10 — Failure case

```bash
python3 evaluation/scripts/run_stress_tests.py
```
Point out scenario D (heavy fog): **NOT_RECONSTRUCTABLE, score 15.0**, reason: "fewer
than 2 line segments on one or both lane edges."

> "Given footage that genuinely can't support a measurement, CARAI says so — explicitly,
> with a reason — instead of guessing."

## 3:10–4:00 — Evaluation

Screen: `docs/uncertainty_calibration.md` and `docs/erratic_driving_detection.md` side by
side, or read the numbers directly:

- Roll correction: 45.3% → 0.03% speed error (validated scenario)
- Uncertainty coverage: 94.7% against a 90% target, measured across 19 independent trials
- Multi-camera cross-examination: 6/6 scenarios succeeded, 100% direction-sign detection,
  2.95m mean position error after reconciling two independently-calibrated cameras
- Continuous-driving detection: 100% recall on obvious erratic behavior, 0 false positives
  on ordinary driving, degrading gracefully (50%) on marginal cases — not a single
  inflated number, the honest dose-response curve

> "Every number here is measured, including the ones that aren't perfect."

## 4:00–4:30 — Architecture

Screen: `infrastructure/aws/template.yaml` briefly, or an architecture diagram if you
make one (S3 → Lambda container → DynamoDB → API Gateway). If the live AWS deployment
worked, show the real `curl` round trip from `docs/aws_deployment_runbook.md`:

```bash
curl -X POST "$API_ENDPOINT/jobs" -d '{"filename": "clip.mp4"}'
curl -X PUT "<upload_url>" --data-binary @clip.mp4
curl "$API_ENDPOINT/jobs/<job_id>"
```

## 4:30–5:00 — Close

Return to the web app's hero: **"Was it avoidable?"**

> "CARAI doesn't just calculate what happened. It checks its own answer before it
> tells you — and it knows when the footage doesn't have enough evidence to answer
> at all."

## Two additional beats worth including if time allows (cut from the main 5 if tight)

**Multi-camera cross-examination** — the **Multi-Camera Cross-Examination** tab in the
web app, showing two independently-positioned, unsynchronized cameras converging on
the same speed (21.2 vs 21.8 m/s) with no shared clock or coordinate system fed in.

**Continuous monitoring** — the **Continuous Monitoring** tab's timeline, showing
ground truth vs. detected erratic-driving intervals side by side, including the honest
miss (not hidden) on the smaller-amplitude segment.

## What NOT to do

Don't round any number up. Don't cut the parts where CARAI is wrong or uncertain — the
Skeptic beat and the honest evaluation numbers are the strongest material here, not a
liability to downplay. A judge who reads the docs will find the same numbers; the video
should match them exactly.
