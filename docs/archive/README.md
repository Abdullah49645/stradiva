# Archive: v1 documentation (as received)

These documents describe CARAI v1, the repository as it was received. They
are kept unchanged for the record. **Several of their claims did not hold up
when re-tested.** The table gives what re-testing found. Current
documentation is in the parent folder, and every number there is
reproducible from `evaluation/`.

| v1 document | v1 claim | What re-testing found |
|---|---|---|
| week2_assessment.md | Roll correction: 0.03 % speed error (vs 45.3 % uncorrected) | Holds for one scene whose roll (8.0°) the integer-degree search grid hits exactly. On simulator v2 with non-integer roll, yaw and realistic 15 cm paint, the v1 geometry accepted 4 of 18 scenes, with 3.7–9.3 % error. In one scene a 4° yaw was "fixed" by a −9° roll (9.3 % error) and the v1 Skeptic stayed silent (`evaluation/results/baseline_legacy_geometry_sweep.json`). Even the unrolled scene A had 2.24 %. |
| skeptic_agent.md | The Skeptic catches a regression that turned 0.03 % into 91 % | Reproduced for that one reverted bug. But the v1 checks re-use the fit's own data: on the original demo it also **disputed a correct reconstruction** (3.6 % error). There was no fault-injection benchmark. v2 replaces it (backend/verification/skeptic.py; docs/EVALUATION.md). |
| uncertainty_calibration.md | "Calibrated" 90 % intervals | The scale factor (2.5) was fitted and evaluated on the same 19 trials. A rerun recommends 1.25, and coverage is 18/19 at any factor ≥ 1.25, so the evidence does not distinguish them. This is in-sample calibration, not calibration. v2 reports coverage only on held-out data. |
| week3_4_causality_assessment.md, report_*.md | Demo verdicts (avoidable / unavoidable) | The production path (`backend/api/process_video.py`) issues **no verdict** on any of the three demo collisions: it tracks nothing on two and finds no collision on the third. The UI's verdicts came from an evaluation script that read the ground-truth collision frame and used a background rendered from ground truth. Demo case 002's displayed verdict (NOT_AVOIDABLE) contradicts its own ground truth (avoidable). |
| multi_camera_architecture.md, multi_camera_*.json | Two-camera sync to 0.017 s | That is the proof-of-concept number. The sweep in the same folder has a mean sync error of 0.41 s, and the cross-examination script crashes. **De-scoped (experimental)**; not part of v2. |
| erratic_driving_detection.md | 67 % recall / 50 % precision | Not re-validated; it uses a separate synthetic set. **De-scoped (experimental)**; not part of v2. |
| cool_graviton_assessment.md | No Graviton benchmark run | Still true. COOL is not entered. |
| aws_and_production_notes.md, aws_deployment_runbook.md | Template + moto tests; not deployed | Superseded by docs/aws.md (template rewritten: CORS, idempotency, sizing, retention). Still not deployed. |
| definition_of_done.md, priorities.md, week1_feasibility.md, demo_video_script.md, agent_tool_upgrades.md | Plans and feasibility notes | Historical. |
