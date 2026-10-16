# Demo plan: 5-minute video, with a backup

Goal: in five minutes, show a judge three things. CARAI measures motion
from markings with OpenCV. It refuses when the footage cannot support a
claim. Every claim is tested, failures included. Nothing in the video may
suggest the collision verdicts were validated on real crashes, because they
were not.

## Script (target 4:40, hard limit 5:00)

| Time | Screen | Say (paraphrase) |
|---|---|---|
| 0:00–0:25 | Case file "Rear-end, roadside camera", evidence player paused on frame 1 | Problem: there is lots of accident footage, but no calibration. Who was going how fast, and could braking have prevented it? CARAI answers from physics only. It never assigns fault. |
| 0:25–1:10 | Play the clip with the road-grid overlay on; point at the dashes, then the horizon | The only ruler is the paint: the dash cycle, a stated assumption of 12 m ± 3 %. OpenCV finds the lines, the vanishing point and the horizon (closed form from three parallel lines). The grid you see is the recovered road plane, drawn back onto the video. |
| 1:10–1:50 | Tracks, the motion chart, "What this video establishes" | MOG2, a Kalman filter and Hungarian assignment track the vehicles. Positions are smoothed on the road plane with uncertainty. The lead vanishes behind the follower after about 1 s. That is too little braking to pass the brake-onset test, so CARAI marks the onset "not observed" and measures reaction budgets from the first frame instead of guessing. |
| 1:50–2:40 | Envelope map, key facts, then "Checked against ground truth" | The verdict is not a single number. Across reaction times of 0.75–2.0 s and braking of 5–8 m/s², the map shows where contact was avoided in the Monte Carlo reconstructions. Here: "depends on the assumptions", avoided only with a reaction within 1.16 s of the first frame. Then show the ground-truth table, including the two rows marked "differs": contact 0.27 s early, reaction budget 0.2 s short. Say clearly: this case is synthetic and illustrative. |
| 2:40–3:15 | Case "Bumped camera": Skeptic panel with temporal stability disputed | The Skeptic re-derives the geometry from each half of the clip. The camera moved, so the verdict is withheld and the number is shown only for review. |
| 3:15–3:35 | Case "No dashed line" | No ruler, no number. CARAI says why. |
| 3:35–4:15 | "Real dashcam footage" tab: comma2k19 holdout chart | Real footage, pre-registered. Our first holdout failed: a city street with the freeway prior gave speeds 2.1× too high. Show it. Then the fix, applied post hoc: a second ruler (lane width × focal length) disagrees, so CARAI refuses. |
| 4:15–4:40 | "How it was tested" tab | Held-out results, including the harmful ones. With true tracks the reasoning is right in half to two-thirds of cases (fewer on the hard shift split), with few harmful errors. From pixels it mostly declines: the frozen code made 8 harmful errors in 30, all far from the camera. The resolution rule added afterwards cut that to 3 in 30 on a fresh split. No real crash footage could legally be used, so verdicts are validated on simulation only. Close on the AWS diagram, one sentence. |

Recording notes: 1920×1080, browser at 110 % zoom, cursor highlight on.
Record the UI from the static build (`npm run build && npx serve out`) so
every number on screen comes from the committed JSON. Record the live
upload separately (below) and cut it in only if it works.

## Live segment (optional, if a deployed endpoint exists)

Upload `evaluation/_sim_cache/demo/demo_rear_end.mp4` (regenerate it with `python3 evaluation/demo/build_cases.py`) in the "Analyse your
own clip" panel. Show the job going `PENDING → PROCESSING → RECONSTRUCTED_COLLISION`.
Processing should take under a minute on Lambda (estimated from local
measurements, see docs/aws.md), so cut the wait in the edit.

## Backup plan

| Failure | Backup |
|---|---|
| No AWS endpoint by recording day | Run the local API (`uvicorn backend.api.server:app`), state on screen "running locally; same job contract as the AWS stack", and show the SAM template and moto tests for ~10 s. Do not imply a live deployment. |
| Live upload errors or is slow | Use the pre-built case files only (they are the same pipeline's output, committed with their inputs). |
| Browser video codec issue (VP8 webm) | Each case JSON renders without video. The motion chart, envelope map and Skeptic panels carry the story. Alternatively, screen-record from Chromium, which plays VP8 natively. |
| Over time | Drop the "No dashed line" case (20 s). |
| A judge asks for a live screen-share | Same flow from the static build, plus a terminal run of `python3 -m pytest -q tests` and `python3 evaluation/experiments/summarize_suite.py`. |

## What not to say

* Not "validated on real crashes", "calibrated intervals" for the collision
  verdicts, or "agentic AI". The loop is deterministic, so call it a
  "decision loop with an independent verifier".
* Not "novel calibration". The individual techniques are established
  (docs/prior_art.md).
* Not "deployed on AWS" unless the runbook has actually been executed.
