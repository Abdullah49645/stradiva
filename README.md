# CARAI: physics from uncalibrated road footage

*Codename, not the final project name.* OpenCV AI Competition 2026 entry.

CARAI measures vehicle motion from ordinary, uncalibrated road footage. The
only ruler it uses is the road's own paint. When two vehicles in the same
lane collide, it states whether braking could have prevented contact, over
an explicit range of reaction times and braking capabilities, and which
assumption decides the answer. It never assigns fault. When the footage
cannot support a claim, it says why instead of giving a number.

**Read first:**
* docs/technical_report.md: problem, method, OpenCV 5, AWS, limitations, responsible use.
* docs/EVALUATION.md: how it was tested.
* docs/EVALUATION_RESULTS.md: every number, generated from `evaluation/results/`.

## What is validated, and what is not

| Claim | Evidence | Status |
|---|---|---|
| Road geometry from markings (vanishing point, closed-form horizon/roll, homography with bootstrap) | Simulator ground truth, 18-scene sweep vs v1 | Validated on simulation |
| Collision verdicts (avoidable / not avoidable / depends) and refusals | Held-out test and shift splits (frozen v2.0) and test2 (v2.1), 3 layers | **Simulation only.** No legally usable real crash footage with ground truth was found. |
| Dashcam ego speed from markings | Two real datasets, pre-registered | **Real footage.** Freeway works; a pre-registered city holdout **failed** (wrong marking prior), and a post-hoc second ruler now refuses that case. |
| Skeptic (cross-examination) | Fault-injection benchmark with negative controls | Measured detection and false-alarm rates |
| Investigator (remedies after a dispute) | test3 (new held-out split) and held-out fault scenes, pre-registered; outcome before vs after on the same clips | Simulation only; refusals turned correct vs turned harmful are both reported |
| Near-miss measures (gap, time to collision, braking needed) | Simulator ground truth over the observed interval | Simulation only |
| Dashcam: distance and closing speed of the vehicle ahead (YOLOX-S on `cv2.dnn`) | The car's own radar, pre-registered, test part run once | **Real footage**, one 60 s segment |
| AWS stack | moto-mocked tests, cfn-lint | **Not deployed** from this environment (docs/aws.md) |
| Multi-camera, erratic driving (from v1) | — | **Experimental, de-scoped** (docs/archive/README.md) |

## Run it

```bash
# Python 3.12+; OpenCV 5 headless, NumPy, SciPy (pinned)
pip install -r requirements-dev.txt

python3 -m pytest -q tests                       # unit, integration, AWS handlers (moto)

bash models/fetch_models.sh                      # vehicle detector for the dashcam lead-vehicle mode (SHA-256 checked)

# analyse a clip from the command line (pipeline + investigator, as the service runs it)
python3 -c "from backend.investigator.investigator import investigate; import json; \
print(json.dumps(investigate('clip.mp4', user_inputs={'fps': 30}), indent=1)[:3000])"

# local API (same job contract as AWS) + web UI
uvicorn backend.api.server:app --port 8000
cd frontend && npm ci && NEXT_PUBLIC_CARAI_API_BASE=http://localhost:8000 npm run dev   # http://localhost:3000
```

The UI ships with case files produced by the pipeline
(`evaluation/demo/build_cases.py`). It works without the API; the API is
needed only to analyse your own clip.

To rebuild every evaluation number, see docs/EVALUATION.md §8. To deploy
to AWS, see docs/aws.md.

## Repository map

| Path | What |
|---|---|
| `backend/pipeline/analyze.py` | The v2 pipeline (ingest → geometry → tracking → kinematics → events → counterfactual → Skeptic) |
| `backend/geometry`, `cv/geometry` | Road geometry from markings |
| `cv/tracking/mot.py` | Background, foreground and multi-object tracking |
| `backend/reconstruction`, `backend/causality` | Smoothing, impact and brake-onset detection, avoidance envelope |
| `backend/verification/skeptic.py` | Cross-examination checks |
| `backend/investigator/` | Remedies for blocking disputes, trace, requests for user inputs |
| `backend/dashcam/lead_vehicle.py`, `cv/detection/` | Vehicle ahead: YOLOX-S on OpenCV DNN, two range rulers, closing speed, time to collision |
| `backend/dashcam/ego_speed.py` | Moving-camera ego speed |
| `models/fetch_models.sh` | Downloads the YOLOX-S ONNX model (OpenCV Model Zoo, Apache-2.0) |
| `backend/api/server.py`, `infrastructure/` | Local API; AWS SAM template, Lambda handlers, Dockerfile |
| `evaluation/` | Simulator, suites, real-footage runs, results, report builder |
| `frontend/` | Next.js case viewer and evidence pages |
| `docs/` | Report, evaluation, architecture, AWS, assumptions, prior art, demo plan, licensing, provenance |

v1 modules (`backend/agent`, `backend/api/process_video.py`,
`backend/multiview`, `backend/behavior` and others) are kept only for the
before/after comparison.

## Licence and data

Code: Apache-2.0 (`LICENSE`, `NOTICE`; see docs/licensing.md for the
reasoning). Data: docs/data_provenance.md. One comma2k19 excerpt (MIT) is
bundled. The comma.ai speedchallenge video was used for evaluation only and
is not redistributed. All other footage is synthetic.

Author: [Abdullah49645](https://github.com/Abdullah49645).
