# Demo plan: 5-minute video, with a backup

Goal: in five minutes, show a judge four things. Odolite measures motion from
markings with OpenCV. On real dashcam footage it measures the vehicle ahead
with an OpenCV DNN detector and agrees with the car's own radar. When its
own cross-examination disputes a result, it tries the remedy for that
specific failure and says plainly when the refusal stands. Every claim is
tested, failures included. Nothing in the video may suggest the collision
verdicts were validated on real crashes, because they were not.

The video opens on real footage, because that is where the strongest
validated result is.

## Script (target 4:45, hard limit 5:00)

| Time | Screen | Say (paraphrase) |
|---|---|---|
| 0:00–0:20 | Case "Real dashcam, wrong assumption caught", video playing | Problem: road footage is everywhere, but nobody calibrated the camera. Odolite turns it into measurements from physics only, and it never assigns fault. This is one minute of real driving from the comma2k19 dataset. |
| 0:20–1:10 | Scroll to "Vehicle ahead: distance and closing speed" | YOLOX-S running on OpenCV 5's DNN module finds the car in our lane. Its distance comes from two independent rulers: where its tyres meet the road, and how wide it looks. If they disagree, Odolite refuses the window. Grey is the car's own radar, which Odolite never saw. Pre-registered: the first 20 s for development, the last 40 s run once. Range within 2.4 % median, every interval contains the radar; closing speed within 0.6 m/s. |
| 1:10–1:40 | Scroll up to own-vehicle speed and the refusal table | The same footage under a freeway marking assumption: the lane width and the camera's focal length contradict it in 19 windows, so Odolite answers none instead of giving speeds twice too high. This was our pre-registered holdout, and its first version failed. The "Real dashcam footage" tab shows that failure. |
| 1:40–2:25 | Case "Rear-end, roadside camera": play with the road-grid overlay | Static camera: the only ruler is the paint (a stated 12 m ± 3 % dash cycle). Lines, vanishing point and a closed-form horizon give the road plane, drawn back onto the video. MOG2, Kalman and Hungarian track the vehicles. |
| 2:25–3:05 | Envelope map, "How close they came", then "Checked against ground truth" | The verdict holds across reaction times 0.75–2.0 s and braking 5–8 m/s², or it says which assumption decides it. Here it depends. The near-miss measures (smallest gap, time to collision, braking needed) are the standard traffic-conflict figures. Show the ground-truth rows marked "differs" too. This case is synthetic and illustrative. |
| 3:05–3:50 | Case "Camera knocked mid-clip": "What Odolite tried after the dispute" | The cross-examination finds the camera moved. Odolite locates the bump (ECC alignment), splits the clip and re-analyses each part; each part must pass the whole cross-examination again. Here neither does, so the refusal stands, and Odolite says what input would help. |
| 3:50–4:10 | Case "Unfamiliar road markings" | The dashes are 6 m + 12 m. Odolite notices the proportions, tries every documented standard that matches, and gives a verdict only if all of them agree. They do not, so no verdict. |
| 4:10–4:45 | "How it was tested" tab: test3 and investigator tables | Held-out, pre-registered, run once. With true tracks the reasoning makes no harmful errors; from pixels it makes 5 in 30, and we show where they come from. The investigator changed no outcome on the fresh split and made one borderline answer on the fault scenes. We publish that. Close on the AWS diagram, one sentence. |

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
