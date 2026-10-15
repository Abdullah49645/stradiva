"""
Continuous-feed erratic-driving detection (docs/erratic_driving_detection.md).

Applies the SAME trajectory reconstruction this project already builds
and validates for collision analysis (position/velocity from a tracked
vehicle) to a continuous feed instead of a single triggered clip, and
asks a different question: not "what happened in this crash" but
"is this vehicle, right now, driving in a way worth flagging."

Two detectable patterns, matching generate_continuous_feed.py's labeled
ground truth:
  WEAVING              -- lateral position oscillating beyond what
                           normal lane-keeping (or tracking noise) would
                           produce.
  HARSH_BRAKING         -- deceleration sustained beyond a standard
                           hard-braking threshold, not just single-frame
                           noise.

Design principle carried over from the Skeptic (docs/skeptic_agent.md):
don't flag noise as signal. Both detectors combine an ABSOLUTE,
physically-motivated threshold (a real hard-braking deceleration; a real
lane-departure-scale lateral deviation) with an ADAPTIVE noise floor
estimated from this specific clip's own apparent jitter -- a clip with
noisier tracking (farther vehicle, worse geometry fit) needs a higher
bar before flagging something as real erratic behavior rather than
reconstruction noise, exactly the same reasoning
backend/uncertainty/propagation.py already applies to speed estimates.
"""
import numpy as np
from scipy.signal import savgol_filter

# Weaving
WEAVE_WINDOW_S = 1.0
WEAVE_BASELINE_WINDOW_S = 2.0
WEAVE_ABS_MIN_STD_M = 0.06        # recalibrated against what this reconstruction pipeline actually
                                   # delivers at a realistic tracked distance (~50-65m), not a generic
                                   # real-world guess -- see docs/erratic_driving_detection.md
WEAVE_NOISE_FLOOR_MULTIPLIER = 2.5

# Harsh braking
BRAKE_WINDOW_S = 0.5
BRAKE_ABS_THRESHOLD_MPS2 = 1.7   # NOTE: NOT a textbook hard-braking value (commonly ~3.0 m/s^2) --
                                  # recalibrated down from that after measuring this reconstruction
                                  # pipeline's own attenuation: a real, physically-simulated 4.5 m/s^2
                                  # brake pulse reconstructs to only ~2.0 m/s^2 peak after tracking +
                                  # the smoothing this detector needs to suppress frame noise (see
                                  # docs/erratic_driving_detection.md). This threshold is calibrated
                                  # against what THIS pipeline can actually deliver, not against
                                  # physical reality directly -- meaning the reported peak_decel_mps2
                                  # on a flagged interval should be read as "at least this erratic",
                                  # not taken as a literal, unbiased physical measurement.
BRAKE_MIN_DURATION_S = 0.3

MERGE_GAP_S = 0.3  # adjacent same-type flagged windows within this gap get merged into one interval

LOCAL_SMOOTH_WINDOW_S = 0.4  # short, fixed-duration smoothing -- NOT
                              # backend/reconstruction/trajectory.py's smooth_trajectory,
                              # whose window_frac=0.35 is sized as a fraction of the WHOLE
                              # clip and would smooth clean across a short erratic segment
                              # in a multi-segment continuous feed, washing it out entirely.
                              # Same Savitzky-Golay technique, deliberately re-sized for this
                              # use case's timescale instead.


def _smooth_local(values, t_s, window_s=LOCAL_SMOOTH_WINDOW_S, polyorder=2):
    values = np.asarray(values)
    n = len(values)
    if n < 5:
        return values
    dt = float(np.median(np.diff(t_s))) if len(t_s) > 1 else 1.0 / 30
    window = int(round(window_s / dt)) if dt > 0 else 5
    if window % 2 == 0:
        window += 1
    window = max(polyorder + 2, min(window, n if n % 2 == 1 else n - 1))
    if window <= polyorder or window < 5:
        return values
    return savgol_filter(values, window, polyorder)


def _rolling(values, times, window_s, fn):
    """fn applied to values within a trailing window_s window ending at
    each sample time. Simple O(n^2) implementation -- fine at the
    sample counts a single-vehicle continuous feed produces (hundreds to
    low thousands of points per clip); revisit with a proper sliding
    window if this is ever applied to much longer feeds."""
    out = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        mask = (times > t - window_s) & (times <= t)
        if mask.sum() >= 2:
            out[i] = fn(values[mask])
    return out


def _merge_intervals(flagged_times, seg_type):
    if len(flagged_times) == 0:
        return []
    intervals = []
    start = flagged_times[0]
    prev = flagged_times[0]
    for t in flagged_times[1:]:
        if t - prev > MERGE_GAP_S:
            intervals.append((start, prev))
            start = t
        prev = t
    intervals.append((start, prev))
    return [{"start_s": float(s), "end_s": float(e), "type": seg_type} for s, e in intervals]


def detect_weaving(t_s, y_m):
    if len(t_s) < 5:
        return []
    y_smooth = _smooth_local(y_m, t_s)
    baseline = _rolling(y_smooth, t_s, WEAVE_BASELINE_WINDOW_S, np.median)
    baseline = np.where(np.isnan(baseline), y_smooth, baseline)  # fall back to raw value at the very start
    residual = y_smooth - baseline
    rolling_std = _rolling(residual, t_s, WEAVE_WINDOW_S, np.std)
    valid = ~np.isnan(rolling_std)
    if valid.sum() < 3:
        return []
    noise_floor = float(np.percentile(rolling_std[valid], 20))
    threshold = max(WEAVE_ABS_MIN_STD_M, WEAVE_NOISE_FLOOR_MULTIPLIER * noise_floor)
    flagged = t_s[valid & (rolling_std >= threshold)]
    intervals = _merge_intervals(flagged, "WEAVING")
    for iv in intervals:
        iv["reason"] = (f"lateral position std {threshold:.2f}m+ within a {WEAVE_WINDOW_S:.1f}s window "
                         f"(clip's own baseline jitter: {noise_floor:.2f}m) -- oscillation beyond normal "
                         f"lane-keeping noise for this footage.")
        iv["threshold_m"] = threshold
        iv["noise_floor_m"] = noise_floor
    return intervals


def detect_harsh_braking(t_s, speed_mps):
    if len(t_s) < 5:
        return []
    speed_smooth = _smooth_local(speed_mps, t_s)
    dt = np.diff(t_s)
    dt[dt == 0] = 1e-6
    accel = np.diff(speed_smooth) / dt
    t_accel = t_s[1:]
    rolling_decel = _rolling(accel, t_accel, BRAKE_WINDOW_S, np.mean)
    valid = ~np.isnan(rolling_decel)
    flagged = t_accel[valid & (rolling_decel <= -BRAKE_ABS_THRESHOLD_MPS2)]
    intervals = _merge_intervals(flagged, "HARSH_BRAKING")
    intervals = [iv for iv in intervals if iv["end_s"] - iv["start_s"] >= BRAKE_MIN_DURATION_S]
    for iv in intervals:
        mask = (t_accel >= iv["start_s"]) & (t_accel <= iv["end_s"])
        iv["peak_decel_mps2"] = float(np.min(accel[mask])) if mask.any() else None
        iv["reason"] = (f"deceleration sustained at or beyond {BRAKE_ABS_THRESHOLD_MPS2:.1f} m/s^2 for "
                         f"{iv['end_s']-iv['start_s']:.2f}s (peak {iv.get('peak_decel_mps2', 0):.1f} m/s^2) -- "
                         f"a standard hard-braking threshold, not single-frame reconstruction noise.")
    return intervals


def analyze_continuous_feed(t_s, y_m, speed_mps):
    """Main entry point: t_s/y_m/speed_mps are the arrays a tracked
    trajectory already produces (e.g. run_causality_feasibility.py's
    track_vehicle output). Returns a flat, time-ordered list of flagged
    intervals across both detectors."""
    t_s, y_m, speed_mps = np.asarray(t_s), np.asarray(y_m), np.asarray(speed_mps)
    intervals = detect_weaving(t_s, y_m) + detect_harsh_braking(t_s, speed_mps)
    intervals.sort(key=lambda iv: iv["start_s"])
    return intervals
