# Pre-registration: real-footage ego-speed evaluation (comma.ai speed challenge)

Committed **before** any CARAI geometry/speed code was run on this footage and
before its speed labels were compared to anything. The git commit that adds this
file is the timestamp. Anything changed after that commit is listed under
"Deviations" at the bottom, with the reason.

## Footage
- Source: `commaai/speedchallenge` (`data/train.mp4`, `data/train.txt`), fetched from
  GitHub LFS. SHA-256 of `train.mp4`:
  `a9f16228de735cbe65946093287bc32e317bea52d0738b81340bcd28f72ae06f`
  (matches the LFS pointer oid in the repo).
- 20 400 frames, 20 fps, 640x480, forward-facing dashcam, one continuous drive:
  freeway (approx. frames 0-8 999) then city streets (approx. 9 000-20 399).
- License: the repository has **no license file**. The README invites anyone to use the
  data for the challenge. CARAI uses it **for evaluation only**, does not redistribute
  the video, and fetches it with a script. The project owner must confirm this use is
  acceptable under the competition's third-party-data rule before submission.
- Never used to develop or tune CARAI before this file was written.

## Label units (inferred, not documented)
The official README does not state the unit of `train.txt`. Freeway frames with
free-flowing traffic carry labels of 23-28, which is consistent with m/s (83-101 km/h)
and implausible as mph on a flowing freeway. Labels are therefore treated as **m/s**.
This is an inference from scene context, recorded here before any comparison.

## What is measured
The forward speed of the camera vehicle (ego speed), recovered from lane markings
alone. Labels are never used inside the estimator.

## Scale prior (fixed now, from public standards, not from the labels)
The overhead sign at the start of the clip names Bayshore Blvd / Sierra Point Pkwy,
i.e. US-101 in the San Francisco Bay Area, so California (Caltrans) striping applies:
- Primary prior: Caltrans freeway broken lane line, 12 ft stripe + 36 ft gap
  (3.66 m + 10.97 m, cycle 14.63 m), lane width 12 ft (3.66 m).
- Prior uncertainty stated as +/-5 % on the cycle length (paint tolerance and wear).
- Sensitivity analysis (reported, not used for the headline): US federal MUTCD
  10 ft + 30 ft (cycle 12.19 m), the codebase's old default.

## Metrics (fixed now)
On frames where CARAI returns an estimate (does not abstain):
- MAE and median absolute percentage error of speed vs label.
- Empirical coverage of CARAI's stated 90 % interval.
- Abstention rate, split by freeway / city segment.
- Risk-coverage: error of answered windows vs fraction answered.
Windows of 2 s (40 frames), non-overlapping. Labels averaged over the same window.
No window is excluded after looking at its error.

## Results of the pre-registered run (commit 5bcdb61 estimator, unchanged)
See evaluation/results/comma_eval_california_freeway.json. Reported as-is in docs/EVALUATION.md.

## Deviations (after the pre-registered run -- everything below is POST-HOC)
1. Failure analysis of the pre-registered run found 8 confidently wrong windows: 4 on a freeway
   stretch (frames 6880-7000, ~3x high; the period matches concrete-slab joints, ~4.6 m) and 4 on
   city streets (~1.7-2.3x high; city marking standard differs from the freeway prior).
2. Two checks were added to the estimator in response, both physically motivated and with
   tolerances set ONLY on synthetic development clips:
   - marking-proportion (duty-cycle) check: measured dash length / cycle must match the assumed
     standard (0.25) after the blur inflation measured on synthetic clips (+0.035), +-0.06;
   - continuity check: an answered window that jumps > 6 m/s from both answered neighbours within
     4 s is withdrawn.
   Also: frame rate can now be supplied explicitly; its source is reported.
3. Re-running these checks on the speed-challenge video is NOT independent validation (the checks
   were designed after seeing its failures). It is reported only as "post-hoc".
4. Independent holdout, pre-registered here BEFORE running it: see PREREGISTRATION_HOLDOUT.md.

