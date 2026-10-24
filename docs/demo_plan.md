# Demo plan: 5-minute video, with a backup

Goal: in five minutes, show a judge what the product is: collision footage from an ordinary,
uncalibrated camera in, physics out (speeds, brake onset, contact, near-miss measures, and whether
braking could have prevented it), with every assumption stated. Then show that it cross-examines itself,
that the app works live, the added dashcam mode with its real-footage validation, the architecture, and
the held-out results, failures included. Nothing in the video may suggest that the collision verdicts were
validated on real crashes, because they were not.

## Script (4:37; voiceover with timecodes in evaluation/demo/video/voiceover.md)

| Time | Screen | Point |
|---|---|---|
| 0:00–0:13 | Title card (team, one-line product) | What Stradiva is: collision footage in, physics out |
| 0:13–0:48 | Case "Rear-end, roadside camera", the clip with the road grid | The paint is the ruler: vanishing point, closed-form horizon, tracking; speeds, brake onset, contact (synthetic, with ground truth) |
| 0:48–1:26 | Envelope map, "How close they came", "Checked against ground truth" | A verdict only if it holds across the reaction and braking range; near-miss measures; the rows where it is off |
| 1:26–2:08 | Case "Camera knocked mid-clip": the investigator trace | Cross-examination, the remedy (ECC bump detection, re-analysis), the refusal stands, the input request |
| 2:08–2:33 | Live upload with the frame rate filled in (wait cut) | The app working end to end |
| 2:33–3:21 | Case "Real dashcam": the vehicle ahead against radar, then the own-speed refusal | The added mode: YOLOX-S on cv2.dnn, two range rulers, pre-registered real-footage result |
| 3:21–3:49 | Architecture card | OpenCV 5 components; the AWS path: API Gateway, S3, container Lambda, DynamoDB, expiry |
| 3:49–4:27 | "How it was tested": test3, investigator, radar tables | Held-out results with the failures |
| 4:27–4:37 | Closing card | No fault; measures what the footage supports |

Recording notes: 1920×1080, browser at 110 % zoom, cursor highlight on.
Record the UI from the static build (`npm run build && npx serve out`) so
every number on screen comes from the committed JSON. Record the live
upload separately and cut it in only if it works.

## Live segment (optional, if a deployed endpoint exists)

Upload `evaluation/_sim_cache/demo/demo_rear_end.mp4` (regenerate it with
`python3 evaluation/demo/build_cases.py`) in "Analyse your own clip", with
the frame rate (30) filled in under "What you know about the camera". Show
the job going `PENDING → PROCESSING → RECONSTRUCTED_COLLISION`. On the
development machine this takes about 30 s (docs/aws.md); cut the wait.

## Backup plan

| Failure | Backup |
|---|---|
| No AWS endpoint by recording day | Run the local API (`uvicorn backend.api.server:app`) and say on screen "running locally; same job contract as the AWS stack". Show the SAM template and the moto tests for ~10 s. Do not imply a live deployment. |
| Live upload errors or is slow | Use the bundled case files only (the same pipeline's output, committed with their inputs). |
| Browser video codec issue (VP8 webm) | Each case renders without video; the charts and panels carry the story. Chromium plays VP8 natively. |
| Over time | Drop "Unfamiliar road markings" (20 s). |
| A judge asks for a live screen-share | Same flow from the static build, plus `python3 -m pytest -q tests` and `python3 evaluation/experiments/summarize_suite.py`. |

## What not to say

* Not "validated on real crashes", and not "calibrated intervals" for the
  collision verdicts.
* Not "AI agent" or "agentic AI" without qualification. The investigator is
  a deterministic decision loop: hand-written remedies, one per check.
* Not "novel calibration" or "novel detector". The individual techniques
  are established (docs/prior_art.md).
* Not "the investigator improves accuracy". On the held-out data it did
  not; its value is the trace and the input requests.
* Not "deployed on AWS" unless the runbook has actually been executed.

## The recorded video

`evaluation/demo/video/record_demo.py` records the video above from the real UI (static build plus the
local API for the live upload) and cuts it with ffmpeg: 1920×1080, 4:37. The voiceover script with
timecodes is `evaluation/demo/video/voiceover.md`; the title, architecture and closing cards are the HTML
files next to it.
