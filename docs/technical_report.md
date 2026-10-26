# Stradiva: physics from uncalibrated road footage, with stated assumptions

Technical report for the OpenCV AI Competition 2026. Every number in this
report is generated from files in `evaluation/results/`. The commands that
regenerate them are in docs/EVALUATION.md.

## 1. Problem

Collisions are increasingly recorded, by traffic cameras, shop CCTV and
dashcams. That footage is rarely turned into measurements. The camera is
uncalibrated, nobody surveyed the scene, and the questions people need
answered are physical: how fast was each vehicle going? When did the lead
vehicle start braking? Could the following driver have stopped in time?
Today those answers come from a reconstruction expert, days later, or from
someone eyeballing the video.

Stradiva measures what the footage can support and states the rest as
assumptions. It never assigns fault. "Avoidable" in Stradiva means: under the
stated range of reaction times and braking capabilities, braking would have
prevented contact in at least 95 % of the reconstructions consistent with
the video. It is a statement about physics, not about blame.

## 2. Users

* **Accident reconstruction engineers and insurers' technical staff.** They
  get a first quantitative read of footage in minutes, with the assumptions
  spelled out for them to challenge.
* **Road-safety analysts at public agencies.** They can measure speeds and
  near-misses from existing camera footage without calibrating each camera.
* **Drivers with dashcam footage.** They get their own speed from the
  footage (a separate, simpler mode).

None of these users should take a Stradiva number on trust. The product is
built around showing why each number is what it is, and what would change
it.

## 3. What the footage can and cannot establish

A static camera looking along a road sees painted markings. Three ideas
from projective geometry give a metric road plane from them, with no
calibration:

1. **Vanishing point and horizon.** The edge lines and the centre line meet
   at the road's vanishing point. Because the three lines are equally
   spaced, the centre line is the harmonic conjugate of the edges with
   respect to the horizon direction. That gives the horizon, and hence the
   camera roll, in closed form (`cv/geometry/projective.py`).
2. **Inverse distance is affine along a line.** For any camera pose, along
   one 3D line the inverse image distance to its vanishing point, s = 1/d,
   is affine in road distance. Sampling a strip uniformly in s
   (`cv2.remap`) turns the perspective-compressed dashes into an evenly
   spaced pattern. Fitting the dash centres gives the affine map, which
   fixes the along-road scale *given the dash cycle length*.
3. **A homography from four constraints.** The vanishing point, the horizon,
   the lateral spacing and the along-road map determine the image-to-road
   homography. Bootstrapping the dash fit gives a distribution of
   homographies, not just one.

The dash cycle length (dash plus gap, 12 m by default) sets the speed
scale. **It cannot be identified from a single static view**, because a
road with longer dashes seen from farther away looks the same. Stradiva
therefore treats it as a stated prior with ±3 % uncertainty, reports it
on every result, and tests its *proportions* (duty cycle), which are
observable. docs/assumptions.md lists every such assumption.

## 4. Architecture

See docs/architecture.md for the diagrams. The pipeline
(`backend/pipeline/analyze.py`):

1. **Ingest.** Decode, cap at 20 s and 1280 px, hash the file.
2. **Camera-motion test.** Lucas–Kanade features with a forward-backward
   check. A moving camera is routed to the dashcam estimator.
3. **Background.** A temporal median, refined twice by excluding tracked
   vehicles. A vehicle that stops (for example after a collision) stays
   excluded where it was last seen.
4. **Road geometry**, as in section 3, from the plain median background.
5. **Tracking.** MOG2 foreground with shadows removed, then Kalman
   prediction and Hungarian assignment (SORT-style). A merged blob keeps
   measuring the nearer vehicle's ground contact (still visible) and marks
   the hidden one OCCLUDED. Truncated boxes are detected. The ground contact
   is the lowest solid row of the unclosed mask.
6. **Kinematics.** Ground contacts go through each bootstrap homography and
   a Rauch–Tung–Striebel constant-acceleration smoother, with per-point
   measurement noise propagated from pixels through the homography Jacobian.
   Tracklets are stitched in road coordinates.
7. **Events.** Find a same-lane closing pair. The lead's brake onset comes
   from a hinge regression, accepted only if a likelihood-ratio test against
   constant acceleration passes. The impact comes from a sudden speed change
   (a velocity kink) of whichever vehicle stays visible, or of the gap
   between the two. Without an impact signature, contact is extrapolated
   across occlusion, and that contact must survive the vehicle-length
   uncertainty. A "no contact" claim must survive it too, and must not rest
   on vehicles that left view while closing. Otherwise the result is
   CONTACT_UNDETERMINED.
8. **Counterfactual envelope.** Monte Carlo over the homography bootstrap,
   the marking prior, vehicle length, brake-onset interval and the hidden
   vehicle's deceleration (when an impact is measured, constrained to meet
   it). Each sample is replayed with the follower braking at a after reaction
   time r, over r in 0.75–2.5 s and a in 4–8 m/s². The verdict holds only if
   it holds over the whole envelope (0.75–2.0 s, 5–8 m/s²). Otherwise
   Stradiva reports the critical reaction time with its 90 % interval.
9. **Skeptic.** It cross-examines the result with evidence the fit did not
   use: leave-one-dash-out prediction, geometry re-estimated on each half of
   the clip, duty cycle against the assumed standard, vehicle footprint
   widths through the homography, and physical plausibility of the motion.
   A blocking dispute withholds the verdict, and the computed result is
   still shown for review.
10. **Investigator** (`backend/investigator/investigator.py`). A blocking
   dispute names a failure, and each failure admits one remedy: a camera
   that moved mid-clip is located by ECC image alignment and each stable
   part is analysed on its own; marking proportions that do not match the
   assumed standard trigger a re-run under every documented standard with
   matching proportions (the verdict must hold under all of them, and speed
   intervals become their union); a subject vehicle's physically impossible
   motion at specific frames is a tracking failure, so those measurements are
   quarantined and the reasoning re-run. The remedy's result goes through the
   whole cross-examination again and replaces the refusal only if every
   blocking check holds and a same-lane pair is reconstructed. Inputs the
   footage cannot provide (frame rate, focal length, marking standard) are
   returned as explicit requests, and the API accepts them with the upload.
11. **Near-miss measures.** For every reconstructed pair: smallest bumper
   gap, shortest time to collision and the largest deceleration that would
   have been needed to avoid contact (DRAC), each with a 90 % interval, over
   the interval in which both vehicles were measured. These are the standard
   traffic-conflict measures road-safety analysts use, and they need no
   collision to be useful.

**Dashcam mode** (`backend/dashcam/`). A moving camera gets two
measurements: its own speed from the lane markings sliding past, and, when
the focal length is known, the vehicle ahead. YOLOX-S (OpenCV Model Zoo)
on `cv2.dnn` finds the vehicle in the ego lane. Its range comes from two
rulers that rest on different assumptions: the flat-road ground plane, with
the camera height derived from the lane width, and the vehicle's width in
pixels against a width prior for its class. They are fused when they agree
and the window is refused when they do not. The closing speed is the range
times the rate at which the vehicle grows in the image, and the time to
collision is reported only while the closing is significant.

## 5. OpenCV 5 implementation

OpenCV 5.0 (`opencv-python-headless==5.0.0.93`) performs every image
operation at runtime:

* **Decoding:** `VideoCapture`
* **Camera-motion test:** `goodFeaturesToTrack`, `calcOpticalFlowPyrLK`
* **Lane lines:** `Canny`, `HoughLinesP`
* **Strips sampled in inverse distance:** `remap`
* **Sub-pixel line refinement:** `fitLine` with Huber weighting
* **Foreground:** `createBackgroundSubtractorMOG2`, `morphologyEx`,
  `connectedComponentsWithStats`, `findContours`
* **Tracking:** `KalmanFilter`
* **Dashcam strip shift:** `phaseCorrelate`, `createHanningWindow`
* **Vehicle detection (dashcam lead vehicle):** `dnn.readNet` (YOLOX-S ONNX
  from the OpenCV Model Zoo), `dnn.NMSBoxes`
* **Camera-bump detection (investigator):** `findTransformECC` (Euclidean)
* **Mapping:** `perspectiveTransform`, `findHomography`
* **Evidence clips:** `VideoWriter`, VP8

The road-plane mathematics (the harmonic horizon, the affine fit in s, the
homography assembly and its bootstrap) is our own NumPy code on top. SciPy
is used only for the Hungarian assignment.

## 6. AWS delivery

One SAM template (`infrastructure/aws/template.yaml`) defines:

* an API Gateway and an API Lambda that issues presigned upload URLs and
  reports job status;
* a private S3 bucket with lifecycle expiry;
* a DynamoDB jobs table with a TTL;
* a container-image Lambda (900 s; 4 GB by default, 3008 MB on the deployed
  account) that runs the full pipeline on each upload.

Duplicate S3 events are absorbed by a conditional status write. The handlers
are tested against moto mocks. The stack is deployed in `us-east-1`
(arm64 Graviton Lambda) and serves the web UI at
https://stradiva-liard.vercel.app; a job was verified end to end on it.
docs/aws.md gives the endpoint, the settings, the runbook and what was
verified.

## 7. Evaluation

The full method, all numbers and the reproduction commands are in
docs/EVALUATION.md. Summary:

* **Layered.** Geometry is tested against the simulator's exact
  homography. The reasoning layer is tested with tracks built from ground
  truth (oracle_clean, and oracle_occluded, where the farther vehicle is
  hidden whenever the nearer one covers it). The full pipeline is tested
  end to end. Each layer's errors are separated from the next layer's.
* **Splits.** Seeds for dev, test and shift were fixed in advance. Code was
  developed on dev only, then frozen. Test and shift were run once. Shift
  over-samples the camera families development found hardest.
* **Outcome categories** punish confident errors, not abstention: correct,
  correct abstention, abstained, missed collision, false collision, wrong
  contact time, overconfident, catastrophic.
* **Real footage.** Dashcam ego speed was tested on two real datasets under
  written pre-registrations, including a holdout that **failed**. That
  failure is reported. No legally usable real collision footage with ground
  truth was found, so **the collision verdicts are validated on simulation
  only**.

Headline results (all from EVALUATION_RESULTS.md; 30 scenes per cell;
"harmful" = missed collision, false collision, verdict on the wrong contact
time, overconfident or opposite verdict):

| | Reasoning layer, true tracks | Reasoning layer, true tracks + occlusion | Full pipeline from pixels |
|---|---|---|---|
| test, frozen v2.0: correct / harmful | 20 / 2 | 15 / 6 | 4 / 8 |
| shift, frozen v2.0 | 8 / 0 | 7 / 6 | 5 / 2 |
| test2 (new held-out), v2.1 | 15 / 0 | 13 / 1 | 5 / 3 |
| **test3 (new held-out), shipped v2.2 + investigator** | **9 / 0** | **9 / 1** | **5 / 5** |
| v1 as received, same clips (full pipeline) | – | – | 0 verdicts issued on test |

What these say:
* The reasoning layer is sound when it gets good tracks.
* The full pipeline mostly declines. All 8 harmful frozen errors on test
  occurred 70–130 m from the camera, which motivated v2.1's resolution rule.
  On the new split, v2.1 cut harmful errors from 8 to 3, with correct
  answers about the same (4 → 5).
* The remaining errors come from tracking. Detail: EVALUATION_RESULTS.md,
  "Where the full pipeline fails".
* v2.1 trades coverage for safety. Re-run post hoc on test, harmful errors
  fall from 8 to 1 end to end and from 6 to 0 with true tracks plus
  occlusion. But correct answers in the occluded layer also fall, from 15
  to 7, because most test events happen far from the camera, where v2.1 now
  declines.

What test3 adds (pre-registered, run once on the code tagged `eval-freeze-v22`):
* The end-to-end harmful count did not improve on a fresh split (5/30;
  test2 gave 3/30 for v2.1). Four of the five rest on a contact
  extrapolated while the follower was hidden, in approaching traffic; with
  true, unoccluded tracks the same clips give none. This is the next
  failure family to fix, and it was found on held-out data.
* The investigator blocked nothing new and recovered nothing: it acted on
  the 10 disputed clips and in every one concluded that the dispute stands
  (6 times because the impossible motion was next to the possible impact).
  On the held-out fault scenes it recovered one answer, whose worst speed
  error was 12 %, just over the 10 % line, so it counts as harmful. Its
  value today is the trace and the input requests, not extra verdicts.
* Near-miss measures are accurate with true tracks (smallest gap within
  0.3-0.6 m median, intervals covering 13 of 13) and not yet reliable from
  pixels (1 of 4 intervals covered).
* **The vehicle ahead, on real footage against the car's radar**
  (pre-registered, test part run once): range median error 2.4 % with all
  17 answered windows inside their 90 % interval, closing speed median
  error 0.58 m/s with 82 % coverage, and time to collision inside the
  interval in the 3 windows where both closed.

Other results:
* **Road geometry**, same 18 scenes: v1 accepted 4 with up to 9.3 % error;
  v2 accepted 18 with at most 0.23 %.
* **Dashcam on real freeway footage** (pre-registered): median error 2.3 %,
  90 % interval coverage 90 %.
* **Real holdout** (pre-registered): **failed**, about 2× too fast because
  the marking prior was wrong for a city street. A post-hoc second ruler
  now makes Stradiva refuse that case.

## 8. Limitations

* **Simulation-only verdicts.** The simulator renders box-shaped vehicles
  with simple shading. Real vehicles, shadows, lighting, compression and
  occlusion by other objects are harder. End-to-end numbers on simulation
  are an upper bound on what to expect from real footage.
* **Tracking is the weakest layer.** The reasoning layer with ground-truth
  tracks does much better than end to end (EVALUATION.md, section 2). Small
  or merged vehicles at long range and vehicles entering already fused are
  the main failures.
* **Range.** Beyond roughly 70 m from the camera, one image row spans
  0.7 m of road or more. Contact then cannot be established within the
  vehicle-length uncertainty. Since v2.1, Stradiva says so instead of
  guessing. Since v2.2 the check is made at the event (the contact, or the
  closest approach), not where the follower was last tracked.
* **Logic gap found on test2, fixed in v2.2 (after test2 was scored, so
  test2 stays clean):** in traffic receding from the camera, an apparent
  "lead pushed forward" speed jump was accepted as an impact signature,
  although there the lead is the hidden vehicle. It is now used only while
  the lead stays in view. Its effect is measured on test3.
* **The investigator is conservative by construction.** It adopts a remedy
  only if every blocking check passes on the result from scratch, so on
  poorly tracked clips it mostly reports that the dispute stands. It cannot
  repair a wrong frame rate or a marking standard with the right proportions
  but the wrong length; those remain requests to the user.
* **Near-miss measures** inherit the tracking errors: they are withheld when
  the reconstruction is disputed or contact is undetermined, and are scored
  against ground truth only where Stradiva reports them.
* **Vehicle ahead (dashcam)** is validated on one real 60 s segment with
  one lead car whose width is close to the 1.8 m prior. It needs the
  camera's focal length.
* **The marking prior is the speed scale.** If the jurisdiction's standard
  is wrong, every speed is wrong by the same factor. The duty-cycle check
  catches standards with different proportions. A standard with the same
  proportions but a different length (US 10/30 ft vs Caltrans 12/36 ft)
  cannot be caught from a static camera. For dashcams with a known focal
  length, the lane-width ruler can catch it.
* **Straight, flat road** near the vehicles, a static camera for collision
  analysis, and a frame rate taken from the container unless the user
  supplies it.
* **Two-vehicle, same-lane, rear-end geometry only.** Angle and side
  impacts, pedestrians and multi-vehicle events are out of scope.
* **Multi-camera fusion and erratic-driving detection** from v1 are
  de-scoped as experimental (docs/archive/README.md).

## 9. Responsible use

* **No fault.** Outputs describe motion and what braking could have
  achieved under stated assumptions. The UI never names a party as at fault.
* **Refusal is a feature.** NOT_RECONSTRUCTABLE, CONTACT_UNDETERMINED and
  DISPUTED are first-class results with reasons, so users are not handed a
  number the footage cannot support.
* **Assumptions travel with results.** Every case states the marking
  prior, vehicle-length prior, reaction-time and braking envelope, and the
  identifiability of each quantity.
* **Not validated for legal use.** Stradiva is a research prototype, and its
  verdicts are validated on simulation only. It must not be the sole basis
  of any legal, insurance or disciplinary decision. Any use in such a
  setting needs an expert who can examine the assumptions.
* **Data.** Uploaded footage is private, expires after 7 days (results
  after 30), and is used for nothing else. Footage of identifiable people
  should be handled under the uploader's local privacy law.
* **Evaluation honesty.** The splits were fixed in advance, real-footage
  tests were pre-registered, failures are published, and no ground truth
  was changed after results were seen. Deviations from the
  pre-registrations are listed with each one.

## 10. What is new, and what is not

Every individual technique is established (docs/prior_art.md). The
contribution is the combination for accident footage, together with the
evaluation discipline:

* verdicts that say which assumption decides them;
* cross-examination whose detection rates are measured by fault injection;
* remedies that must pass the same cross-examination before they replace a
  refusal, evaluated as refusals turned correct against refusals turned
  harmful on held-out clips;
* two independent range rulers for the vehicle ahead, with refusal when
  they disagree;
* a second, independent ruler for metric scale;
* pre-registered real-footage tests that report their failures.
