# Prior art, and what is (and is not) new in CARAI

Short version: every individual technique CARAI uses is established. What we can
defend as a contribution is (a) the combination, applied to accident footage,
and (b) the evaluation discipline around it: assumption-envelope verdicts,
cross-examination with independent evidence measured by fault injection, and
pre-registered real-footage tests that report their failures.

| Technique in CARAI | Status | Closest prior art |
|---|---|---|
| Vanishing point from lane lines (RANSAC over Hough segments) | Established | Textbook (Hartley & Zisserman, *Multiple View Geometry*, ch. 8); standard in traffic-camera calibration |
| Metric scale from lane-marking dimensions | Established | Dailey & Cathey / Schoepflin & Dailey: automatic calibration of roadside cameras from lane-marker periodicity, resampling to remove perspective, autocorrelation (WSDOT research reports, 2000s) |
| Speed from a calibrated homography | Established | Surveyed in Fernández Llorca et al., "Vision-based vehicle speed estimation: a survey", IET ITS 2021 (arXiv 2101.06159); benchmark: BrnoCompSpeed (Sochor et al., IEEE T-ITS 2018, arXiv 1702.06441) |
| Horizon from three equally spaced parallel lines (harmonic conjugate) -> closed-form roll | Established projective geometry; we have not found it used for traffic-camera roll in the papers above | Cross-ratio / harmonic conjugates: Hartley & Zisserman ch. 2 |
| s = 1/d resampling along a line (dash pattern becomes exactly periodic for any camera pose) | A known consequence of the cross-ratio; our use for dashcam ego speed is a specific application | Schoepflin & Dailey resample whole images to remove perspective |
| Multi-object tracking: Kalman + Hungarian | Established | SORT (Bewley et al., ICIP 2016) |
| Ghost / shadow handling in background subtraction | Established | Cucchiara et al., "Detecting moving objects, ghosts and shadows", IEEE PAMI 2003 (tried; disabled -- see technical report) |
| RTS smoothing with heteroscedastic measurement noise | Established | Rauch, Tung & Striebel 1965 |
| Monte Carlo uncertainty in accident reconstruction and avoidance analysis | Established in forensic practice | Wood & O'Riordain, SAE 940720 (1994); SAE 2007-01-0741 on interpreting Monte Carlo reconstructions |
| Abstention / selective prediction | Established | Geifman & El-Yaniv, "Selective classification for deep neural networks", NeurIPS 2017 |

## What we claim, precisely

1. **Verdicts that state which assumption decides them.** Forensic avoidance
   analysis usually evaluates one or a few (reaction time, deceleration)
   pairs. CARAI evaluates the whole envelope on Monte Carlo samples of the
   video reconstruction, and only states AVOIDABLE / NOT_AVOIDABLE when the
   answer holds across the envelope. Otherwise it reports the critical
   reaction time and its interval. This is a new *combination*, not a new
   algorithm.
2. **Cross-examination with evidence the fit did not use, measured.** Checks of
   this kind exist individually. What we add is a fault-injection benchmark
   that measures what each check catches and what none can (e.g. two marking
   standards with the same proportions).
3. **Two independent rulers for metric scale.** With a known focal length, the
   lane width and the dash cycle are separate scale sources. Their
   disagreement detects a wrong marking prior that no single-ruler method can
   see. On the real holdout segment this turned confident 2x errors into
   refusals, and recovered the actual cycle (post-hoc; see EVALUATION.md).
4. **Failure-reporting evaluation.** Pre-registered real-footage tests,
   including the one that failed.

What we do NOT claim: a novel vehicle detector, tracker, calibration method or
speed estimator; state-of-the-art speed accuracy (we have not run BrnoCompSpeed);
real-world validation of the collision verdicts (no labelled real crash
footage exists that we can legally use).
