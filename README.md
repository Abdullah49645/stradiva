# CARAI (codename) — Uncalibrated Accident-Footage Reconstruction

OpenCV AI Competition 2026 submission candidate. Codename only — not final project name.

CARAI reconstructs vehicle motion from ordinary, uncalibrated accident footage — dashcam,
CCTV, phone video, whatever exists — and determines, from the physics alone, whether a
collision appears to have been avoidable. It never assigns fault: only physical facts
(speed, gap, braking, timing) and a stated verdict about what the physics shows, not who's
to blame. See `docs/definition_of_done.md` and `docs/week1_feasibility.md` for the original
scope, and the docs linked below for what's actually been built and validated since.

## What makes this different from a generic CV pipeline

1. **CARAI checks itself.** A second, adversarial agent — the Skeptic — cross-examines
   every accepted geometry/homography using evidence the primary reconstruction never looked
   at. Not hypothetical: `docs/skeptic_agent.md` documents a real near-miss (a code change
   that silently turned 0.03% speed error into 91%) that the Skeptic catches and the original
   pipeline did not.
2. **It knows what it doesn't know.** Every speed estimate ships with a calibrated 90%
   confidence interval, not a bare number — `docs/uncertainty_calibration.md` measures actual
   coverage against synthetic ground truth rather than asserting it.
3. **It knows when to say no.** A dedicated reconstructability score explicitly rejects
   footage that can't support a quantitative claim, with a stated reason — never a confident
   wrong number.

## Status

Past the original Week 1–4 feasibility phases (see `docs/week1_feasibility.md` through
`docs/week3_4_causality_assessment.md`) into an active build-out phase. Current capabilities:

| Capability | Status | Doc |
|---|---|---|
| Camera-roll active correction (not just detection) | Validated, 0.03% speed error | `docs/week2_assessment.md` |
| Uncertainty propagation + calibrated confidence intervals | Validated (coverage measured) | `docs/uncertainty_calibration.md` |
| Agent geometry search (neighbor-frame recovery) | Validated, measured improvement | `docs/agent_tool_upgrades.md` |
| The Skeptic (adversarial cross-examination) | Validated against a real reverted bug | `docs/skeptic_agent.md` |
| Multi-camera cross-examination (2 unsynchronized cameras) | Proof of concept, sync validated | `docs/multi_camera_architecture.md` |
| Continuous-feed erratic-driving detection | Real measured 67% recall / 50% precision | `docs/erratic_driving_detection.md` |
| AWS deployment (Lambda + API Gateway + S3 + DynamoDB) | Template + handlers tested via moto; not yet deployed live | `docs/aws_and_production_notes.md`, `docs/aws_deployment_runbook.md` |
| COOL/Graviton benchmark | Profiled, benchmark ready, no real ARM comparison run yet | `docs/cool_graviton_assessment.md` |
| Real (non-synthetic) footage | Harness built, no clips sourced yet | `evaluation/dataset/real/README.md` |

Every doc above states plainly what's validated versus what's still open — that's
deliberate, not an oversight, and matches the project's own standard for the pipeline itself:
never claim more confidence than the evidence supports.

## Running it

```bash
pip install -r requirements.txt --break-system-packages
pip install pytest moto[s3,dynamodb] boto3 cfn-lint --break-system-packages   # test-only

python3 evaluation/scripts/run_stress_tests.py               # core CV pipeline regression
python3 evaluation/scripts/run_causality_feasibility.py       # collision + Skeptic + uncertainty demo
python3 -m pytest tests/test_aws_handlers.py -v                # AWS logic against mocked AWS

cd frontend && npm install && npm run dev                      # web UI at localhost:3000
```

Full command list, including the multi-camera and erratic-driving evaluation scripts and
real AWS deployment, is in `docs/aws_deployment_runbook.md` and the scripts' own docstrings
under `evaluation/scripts/`.

## Structure

- `backend/` — pipeline stages: video, geometry, calibration, tracking, reconstruction,
  uncertainty, reconstructability, causality, agent (Reconstructor + Skeptic), multiview,
  behavior (erratic-driving), reports
- `cv/` — reusable low-level CV building blocks: lines, vanishing points, homography,
  optical flow, tracking, roll correction
- `evaluation/` — synthetic ground-truth dataset + generators + benchmark/evaluation scripts
- `infrastructure/` — AWS SAM template, Lambda handlers, Dockerfile
- `frontend/` — Next.js web UI (collision reconstruction, multi-camera, continuous
  monitoring views)
- `docs/` — the actual paper trail: what was built, what was measured, what was found and
  fixed, what's still open, for every capability above
