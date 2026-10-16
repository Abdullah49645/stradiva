# Data provenance

Every clip used anywhere in this repository is listed here. Nothing else
was used.

| Data | Source | Licence / rights | Used for | In the repository? |
|---|---|---|---|---|
| Synthetic collision suite (dev / test / shift splits) | Rendered by `evaluation/sim/world.py` from seeds in `evaluation/experiments/collision_suite.py` | Project's own | Development (dev), held-out evaluation (test, shift) | No. Regenerated deterministically into `evaluation/_sim_cache/` (gitignored). |
| Synthetic dashcam scenes | `evaluation/experiments/dashcam_dev.py` | Project's own | Developing the ego-speed estimator | No (regenerated) |
| Synthetic fault-injection scenes | `evaluation/experiments/skeptic_faults.py` | Project's own | Measuring the Skeptic | No (regenerated) |
| Synthetic demo cases | `evaluation/demo/build_cases.py` | Project's own | UI case files | Yes: `frontend/public/cases/*.webm` and `*.json` |
| comma.ai speedchallenge `train.mp4` + `train.txt` (20,400 frames, 20 fps, highway then city; per-frame speed labels) | https://github.com/commaai/speedchallenge | **No licence file** in the repository. Used for evaluation only, after a pre-registration. Not redistributed. **Confirm this use is acceptable before relying on it in the submission.** | Real-footage test of dashcam ego speed (`evaluation/real/PREREGISTRATION.md`) | No. `evaluation/real/fetch_comma.sh` downloads it and checks its SHA-256. |
| comma2k19, chunk "Example_1", route `b0c9d2329ad1606b|2018-08-02--08-34-47`, segment 40 (video.hevc + CAN speed + sensor timestamps) | https://github.com/commaai/comma2k19 (commit 4c7f1a6) | MIT, Copyright (c) 2018 comma.ai (`licenses/comma2k19-MIT.txt`) | Pre-registered holdout test (`evaluation/real/PREREGISTRATION_HOLDOUT.md`); real-footage demo case | Re-encoded excerpt only: `frontend/public/cases/dashcam_real.webm` |

No real crash footage was used. We found no legally usable real collision
footage with ground-truth speeds, and the collision verdicts are evaluated
only on simulation. The UI and the technical report say so wherever the
verdicts appear.

The legacy v1 synthetic demo videos inherited from the received repository
(`evaluation/dataset/`) are project-generated. They are kept only for the
before/after comparison.
