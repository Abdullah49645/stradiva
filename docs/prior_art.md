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
| Surrogate safety measures: time to collision, deceleration rate to avoid the crash, minimum gap | Established in traffic-conflict analysis | Hayward, "Near-miss determination through use of a scale of danger", Highway Research Record 384 (1972); Cooper & Ferguson, DRAC, *Traffic Engineering & Control* (1976); FHWA Surrogate Safety Assessment Model (Gettman & Head, FHWA-RD-03-050, 2003); from video: Saunier & Sayed (TRR, 2000s) |
| Monocular range to the vehicle ahead from the ground plane; time to collision from image scale change | Established | Stein, Mano & Shashua, "Vision-based ACC with a single camera: bounds on range and range rate accuracy", IEEE IV 2003; Dagan et al., "Forward collision warning with a single camera", IEEE IV 2004 |
| Vehicle detection with a COCO-trained one-stage detector | Established; we use YOLOX-S unchanged from the OpenCV Model Zoo | Ge et al., "YOLOX: Exceeding YOLO series in 2021", arXiv 2107.08430 |
| Camera-motion detection by image alignment (ECC) | Established | Evangelidis & Psarakis, "Parametric image alignment using enhanced correlation coefficient maximization", IEEE PAMI 2008 |
| Diagnose-then-repair loops (a failed check selects a corrective action) | Established in model-based diagnosis and robotics | Reiter, "A theory of diagnosis from first principles", *Artificial Intelligence* 1987; our remedies are hand-written rules, one per check |

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
5. **Remedies that must survive the same cross-examination, measured for
   harm.** The investigator's remedies are simple rules. What we claim is the
   discipline: a remedy's result replaces a refusal only if every blocking
   check passes on it from scratch, it never relaxes a threshold, and its
   effect is reported as "refusals turned into correct answers" against
   "refusals turned into harmful answers" on held-out clips.
6. **Two independent range rulers for the vehicle ahead.** Ground-plane range
   and vehicle-width range use different assumptions (flat road and camera
   height vs a vehicle-width prior); disagreement beyond their uncertainty is
   a refusal. Each ruler is established; the refusal rule is ours.

What we do NOT claim: a novel vehicle detector, tracker, calibration method or
speed estimator; state-of-the-art speed accuracy (we have not run BrnoCompSpeed);
real-world validation of the collision verdicts (no labelled real crash
footage exists that we can legally use).
