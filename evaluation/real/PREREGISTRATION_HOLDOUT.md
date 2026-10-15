# Pre-registration: independent holdout (comma2k19 example segment)

Committed before running the post-hoc estimator (with the duty-cycle and continuity checks) on
this footage, and before reading its speed signal.

## Footage
- commaai/comma2k19 (MIT license), `Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/`:
  one minute of I-280 (California) commute driving, forward camera 1164x874, HEVC.
- Never used during CARAI development.
- Frame rate: the HEVC container reports 25 fps, but the segment's own sensor timestamps
  (`global_pose/frame_times`, 1200 frames over 60 s) show 20 fps. The estimator is given
  20 fps explicitly (in a real case the investigator would supply the device's recording rate);
  the run with the unverified container value (25 fps) is reported alongside as a demonstration
  of why the frame-rate source matters.

## Ground truth
CAN-bus vehicle speed (`processed_log/CAN/speed`, m/s), interpolated to frame times and averaged
over each 2 s window.

## Prior
Caltrans freeway broken lane line, 12 ft + 36 ft (cycle 14.63 m, duty 0.25), +-3 % (1 sigma).

## Metrics
Same as PREREGISTRATION.md: answered fraction, median absolute percentage error, 90 % interval
coverage, catastrophic windows (error > 25 %). 30 windows of 2 s.
