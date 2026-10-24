# Assumptions register

Every number below enters Stradiva's results as a stated input, never as a
constant hidden in code. Where the video cannot test an assumption, the
output says so.

## Road markings (metric scale)

| Assumption | Default | Uncertainty used | Testable from the video? |
|---|---|---|---|
| Dash cycle (dash + gap), static-camera road | 12 m (3 m + 9 m) | ±3 % (1σ) | **No**, from a single static view looking along the road. It sets the speed scale. |
| Dash cycle, US federal (MUTCD) | 40 ft = 12.19 m | ±3 % | Proportions (25 % duty) only |
| Dash cycle, California freeway (Caltrans) | 48 ft = 14.63 m (12 ft + 36 ft) | ±3 % | Proportions only; with a known focal length, also via lane width |
| Duty cycle (dash / cycle) | 0.25 | measured inflation +0.035 (blur), tolerance ±0.06 | **Yes**: catches standards with different proportions |
| Edge-to-centre distance (static camera) | 1.85 m | ±5 % | Partly: vehicle widths (gross errors only) |
| Ego-lane width (dashcam scale cross-check) | 3.6 m | ±10 % | Used to *test* the dash cycle when the focal length is known |

The road standard is jurisdiction- and road-class-specific. On real footage we
measured what happens when it is wrong: a freeway prior on a city street gave
speeds 2.1x too high with tight intervals (EVALUATION.md). Stradiva therefore
reports the prior it used on every result and refuses when its checks
contradict it.

## Camera

| Assumption | Where it matters | Testable? |
|---|---|---|
| Static camera (collision pipeline) | background subtraction | Yes: feature motion test routes moving cameras to the dashcam path |
| Square pixels, principal point near centre | scale cross-check only | No |
| Focal length | dashcam scale cross-check | Supplied by the user / device spec; otherwise the cross-check is skipped and says so |
| Frame rate | every speed, linearly | **No.** Container metadata is often wrong (comma2k19: container 25 fps, sensors 20 fps). The source is reported; the user should confirm it. |
| Lines of constant road distance parallel to the horizon (zero yaw) | lateral positions | Error ~ W·tan(yaw); negligible for along-road speed |

## Vehicles and drivers (avoidability)

| Assumption | Envelope | Source / rationale |
|---|---|---|
| Driver perception-reaction time to a lead vehicle's braking | 0.75–2.0 s (grid 0.75–2.5 s) | Green, "How long does it take to stop?", *Transportation Human Factors* 2(3), 2000: 0.7–0.75 s fully expected, ~1.25 s unexpected, ~1.5 s surprise |
| Achievable braking deceleration | 5–8 m/s² (grid 4–8) | Dry asphalt emergency braking ~0.7–0.8 g, wet ~0.4–0.55 g |
| Follower length (gap between rear contacts) | 4.6 ± 0.35 m | Passenger-car range |
| Lead motion after it is hidden / after contact | measured deceleration, continued to rest; with a measured impact, the deceleration that brings its rear to the follower's front at impact | Physical constraint, not a free parameter |
| Hazard onset | the lead's brake onset, only if identifiable (likelihood-ratio test against constant acceleration) | If not identifiable, no AVOIDABLE / NOT_AVOIDABLE verdict is issued |

Stradiva never states who is at fault. "Avoidable under assumption X" is a
statement about physics given X.
