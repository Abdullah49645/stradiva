# Pre-registration: dashcam lead-vehicle range, closing speed and time-to-collision vs radar

Committed before the lead-vehicle estimator (`backend/dashcam/lead_vehicle.py`) exists, and
before any of its output is compared with radar. The git commit adding this file is the
timestamp. Changes after it are listed under "Deviations".

## What was looked at before writing this (disclosed)
- The radar log of the segment was summarised once, to confirm that a lead vehicle exists: the
  nearest in-lane radar target per 4 s window over the whole minute, at 23-64 m forward, with
  relative speeds between -4.4 and +3.9 m/s.
- The OpenCV Zoo YOLOX-S detector was run on 4 frames (0, 300, 600, 1100) to confirm that it
  detects the cars in this footage. It does.
No estimator output was produced or compared with radar.

## Footage and ground truth
- comma2k19 (MIT), `Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40`: 60 s, 1164x874, 20 fps
  (sensor timestamps `global_pose/frame_times`). The same segment as the ego-speed holdout, but a
  different task. The ego-speed estimator is not an input here.
- Ground truth: the car's forward radar (`processed_log/CAN/radar`: forward distance, lateral
  distance, relative speed, track address). Lead = the radar track with the smallest forward
  distance and |lateral| < 1.6 m, matched to each frame time (nearest radar sample within 50 ms).

## Method (fixed now, implemented next)
- Detection: YOLOX-S (COCO car/bus/truck/motorcycle) on OpenCV 5 DNN, every frame.
- Ego lane: lane-line directions about the road vanishing point (the same Hough + RANSAC stage
  as the ego-speed estimator), per 2 s window.
- Lead: the detection whose bottom-centre lies between the ego-lane lines at its row and is
  nearest the camera, followed by box overlap. A new vehicle starts a new segment.
- Range from the flat-road ground plane: Z = h / tan(theta + atan((v_bottom - c_y) / f)).
  Pitch theta comes from the vanishing point. Focal length f = 910 px (comma EON, documented).
  Camera height h is derived from the ego-lane image geometry with a lane-width prior of
  3.6 m +- 10 %, as the median over the clip. The lane-width prior is the only metric assumption.
- Relative speed is dZ/dt from an RTS smoother on Z. Time-to-collision = Z / closing speed,
  reported only while closing faster than 0.5 m/s.
- Uncertainty is Monte Carlo over the lane-width prior and the per-frame row noise of the box
  bottom (2 px).

## Split
- **Development:** t = 0-20 s. The estimator may be tuned here, including the constant
  camera-to-radar range offset: the median of radar minus estimate on this part.
- **Test:** t = 20-60 s. Run once on the frozen estimator. Results are reported whatever they are.

## Metrics (on 2 s windows, test part)
1. Relative speed: median absolute error vs radar (m/s), and coverage of the 90 % interval.
2. Range, after the development offset: median absolute percentage error, and 90 % coverage.
3. Time-to-collision, where both radar and CARAI report closing: agreement within the interval.
4. Answered fraction, and the reasons for abstaining.

## Deviations
(none yet)
