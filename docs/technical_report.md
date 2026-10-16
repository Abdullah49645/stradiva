# CARAI: physics from uncalibrated road footage, with stated assumptions

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

CARAI measures what the footage can support and states the rest as
assumptions. It never assigns fault. "Avoidable" in CARAI means: under the
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

None of these users should take a CARAI number on trust. The product is
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
road with longer dashes seen from farther away looks the same. CARAI
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
   CARAI reports the critical reaction time with its 90 % interval.
9. **Skeptic.** It cross-examines the result with evidence the fit did not
   use: leave-one-dash-out prediction, geometry re-estimated on each half of
   the clip, duty cycle against the assumed standard, vehicle footprint
   widths through the homography, and physical plausibility of the motion.
   A blocking dispute withholds the verdict, and the computed result is
   still shown for review.

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
* a container-image Lambda (4 GB, 900 s) that runs the full pipeline on
  each upload.

Duplicate S3 events are absorbed by a conditional status write. The handlers
are tested against moto mocks. **The stack has not been deployed from this
development environment.** docs/aws.md separates what is verified from
what is not, and gives the runbook.

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

Results are in EVALUATION.md, sections 2–6.

## 8. Limitations

* **Simulation-only verdicts.** The simulator renders box-shaped vehicles
  with simple shading. Real vehicles, shadows, lighting, compression and
  occlusion by other objects are harder. End-to-end numbers on simulation
  are an upper bound on what to expect from real footage.
* **Tracking is the weakest layer.** The reasoning layer with ground-truth
  tracks does much better than end to end (EVALUATION.md, section 2). Small
  or merged vehicles at long range and vehicles entering already fused are
  the main failures.
* **Range.** Beyond roughly 70 m from the camera, one image row spans about
  a metre of road. Contact then cannot be established within the
  vehicle-length uncertainty, and CARAI says so instead of guessing.
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
* **Not validated for legal use.** CARAI is a research prototype, and its
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
* a second, independent ruler for metric scale;
* pre-registered real-footage tests that report their failures.
