"""
Collision Causality & Avoidability Analysis.

Inserted as a core Stradiva output alongside speed/trajectory, not an
optional extra (per the addendum to the master brief). Given the
reconstructed motion of two vehicles (one of which typically strikes the
other), this module answers:

  - pre-impact speed/direction of each vehicle
  - time-to-collision over the observed window
  - whether evasive action (braking) is detectable, and roughly when it began
  - whether the striking vehicle had a physically plausible opportunity
    to avoid the collision, under explicit, stated physical assumptions
  - an avoidable / not-avoidable / insufficient-evidence verdict, gated
    by the same uncertainty system as every other Stradiva output

Stradiva never claims legal fault. Everything here is a physical, evidence-
based assessment ("appears to have had approximately X seconds of
potential avoidance time"), not a determination of who was at fault.

Physical model
---------------
For a vehicle closing on a stationary or slower obstacle, the minimum
distance needed to stop from speed v under reaction time t_r and maximum
deceleration a_max is:

    d_required(v) = v * t_r + v^2 / (2 * a_max)

At any observed instant before impact, if the remaining gap to the
collision point exceeds d_required(v at that instant), the vehicle had a
physically plausible opportunity to stop before impact from that point
onward. The module scans backward from the collision frame to find the
LATEST instant at which this was still true -- the "last possible
avoidance point" -- and reports how long before impact that was.

a_max and t_r are explicit, stated assumptions (not hidden constants);
downstream output must show them. Real deployment should let these vary
by road surface / vehicle class / driver-population priors rather than
use one fixed pair -- flagged as a known simplification below.
"""
from dataclasses import dataclass, field

import numpy as np


DEFAULT_MAX_DECEL_MPS2 = 7.0     # dry-pavement passenger-vehicle emergency braking, a common textbook figure
DEFAULT_REACTION_TIME_S = 1.2    # typical perception-reaction time assumption
BRAKING_DETECT_THRESHOLD_MPS2 = 1.5  # deceleration magnitude that counts as "braking", not noise
BRAKING_MIN_CONSECUTIVE = 3          # frames of sustained deceleration required to call it braking
STEERING_DETECT_THRESHOLD_MPS = 0.6  # lateral speed that counts as active steering, not lane-position noise
STEERING_MIN_CONSECUTIVE = 3         # frames of sustained one-direction lateral speed required
STEERING_MIN_NET_DEVIATION_M = 0.4   # minimum net lateral displacement over that run (rejects jitter)


@dataclass
class CausalityResult:
    verdict: str  # "AVOIDABLE" | "NOT_AVOIDABLE" | "INSUFFICIENT_EVIDENCE"
    narrative: str
    striking_vehicle_speed_at_earliest_observed_mps: float = None
    time_to_collision_at_earliest_observed_s: float = None
    braking_detected: bool = False
    braking_onset_time_before_impact_s: float = None
    steering_detected: bool = False
    steering_onset_time_before_impact_s: float = None
    steering_direction: str = None
    last_possible_avoidance_time_before_impact_s: float = None
    assumptions: dict = field(default_factory=dict)
    confidence: str = "LOW"  # mirrors reconstructability's absolute-position-style confidence labeling
    evidence: list = field(default_factory=list)
    contribution_factors: dict = field(default_factory=dict)


def detect_braking_onset(t_s, speed_mps, decel_threshold=BRAKING_DETECT_THRESHOLD_MPS2,
                          min_consecutive=BRAKING_MIN_CONSECUTIVE):
    """
    Scan for the first sustained deceleration episode. Returns (onset_idx,
    onset_time_s) or (None, None) if no sustained braking is detected.
    """
    accel = np.gradient(speed_mps, t_s)
    decel_mask = accel < -decel_threshold
    run = 0
    for i, is_decel in enumerate(decel_mask):
        run = run + 1 if is_decel else 0
        if run >= min_consecutive:
            onset_idx = i - min_consecutive + 1
            return onset_idx, float(t_s[onset_idx])
    return None, None


def detect_steering_onset(t_s, y_m, lateral_speed_threshold=STEERING_DETECT_THRESHOLD_MPS,
                           min_consecutive=STEERING_MIN_CONSECUTIVE, min_net_deviation_m=STEERING_MIN_NET_DEVIATION_M):
    """
    Scan for a sustained lateral maneuver (lane change / swerve): a run of
    frames where lateral speed (d(lateral position)/dt) exceeds threshold
    in a consistent direction. Requires BOTH sustained lateral speed AND a
    minimum net lateral displacement over the run, so ordinary tracking
    jitter (which reverses sign randomly) doesn't get misread as steering.

    Returns (onset_idx, onset_time_s, direction) or (None, None, None).
    direction is "left" or "right" in the world-Y sign convention used
    throughout this codebase (see scene_generator.py / ground_plane.py).
    """
    if len(t_s) < min_consecutive + 1:
        return None, None, None
    lateral_speed = np.gradient(y_m, t_s)
    sign = np.sign(lateral_speed)
    fast_mask = np.abs(lateral_speed) > lateral_speed_threshold

    run = 0
    run_start = 0
    for i in range(len(fast_mask)):
        if fast_mask[i] and (run == 0 or sign[i] == sign[i - 1]):
            if run == 0:
                run_start = i
            run += 1
        else:
            run = 1 if fast_mask[i] else 0
            run_start = i
        if run >= min_consecutive:
            net_deviation = abs(y_m[i] - y_m[run_start])
            if net_deviation >= min_net_deviation_m:
                direction = "left" if sign[i] > 0 else "right"
                return run_start, float(t_s[run_start]), direction
    return None, None, None


def required_stopping_distance(v_mps, max_decel=DEFAULT_MAX_DECEL_MPS2, reaction_time=DEFAULT_REACTION_TIME_S):
    return v_mps * reaction_time + (v_mps ** 2) / (2 * max_decel)


def safe_following_speed(gap_m, max_decel=DEFAULT_MAX_DECEL_MPS2, reaction_time=DEFAULT_REACTION_TIME_S):
    """
    Inverse of required_stopping_distance: the highest speed at which a
    vehicle could still have stopped within the given gap, under the
    modeled assumptions. Solves 0 = v^2/(2*a) + t_r*v - gap for v>=0.
    """
    if gap_m <= 0:
        return 0.0
    a_coef = 1.0 / (2 * max_decel)
    b_coef = reaction_time
    c_coef = -gap_m
    discriminant = b_coef ** 2 - 4 * a_coef * c_coef
    if discriminant < 0:
        return 0.0
    return max(0.0, (-b_coef + discriminant ** 0.5) / (2 * a_coef))


def classify_deceleration(peak_decel_mps2):
    """
    Purely descriptive physical bucketing of a vehicle's own braking
    intensity -- gradual/moderate/hard -- against commonly-cited braking
    ranges. This characterizes ONE vehicle's motion in isolation; it is
    not a comparison between vehicles and not a fault/blame judgment.
    """
    if peak_decel_mps2 is None:
        return "not observed"
    d = abs(peak_decel_mps2)
    if d < 2.0:
        return "gradual (< 2 m/s^2)"
    if d < 4.5:
        return "moderate (2-4.5 m/s^2)"
    return "hard/sudden (> 4.5 m/s^2)"


def compute_contribution_factors(striking_speed_mps, gap_at_earliest_m, lead_peak_decel_mps2=None,
                                  max_decel=DEFAULT_MAX_DECEL_MPS2, reaction_time=DEFAULT_REACTION_TIME_S):
    """
    Per-vehicle PHYSICAL observations only -- explicitly not a fault or
    responsibility determination (the project owner and the master brief
    both require Stradiva to never make that claim). Each factor describes
    one vehicle's own motion against a physical reference (the modeled
    safe-following speed for the observed gap; a descriptive braking-
    intensity bucket) rather than ranking or blaming either driver.
    """
    safe_speed = safe_following_speed(gap_at_earliest_m, max_decel, reaction_time)
    pct_over_safe_speed = None
    if safe_speed > 0.1:
        pct_over_safe_speed = 100.0 * (striking_speed_mps - safe_speed) / safe_speed

    return {
        "following_vehicle_safe_speed_for_observed_gap_mps": round(safe_speed, 1),
        "following_vehicle_speed_vs_safe_speed_pct": round(pct_over_safe_speed, 0) if pct_over_safe_speed is not None else None,
        "lead_vehicle_deceleration_classification": classify_deceleration(lead_peak_decel_mps2),
        "note": (
            "These are physical observations about each vehicle's own reconstructed motion, "
            "not a fault, responsibility, or blame determination. Stradiva does not identify which "
            "driver caused the collision."
        ),
    }


def find_last_possible_avoidance_point(t_s, speed_mps, gap_to_collision_m,
                                        max_decel=DEFAULT_MAX_DECEL_MPS2,
                                        reaction_time=DEFAULT_REACTION_TIME_S):
    """
    Scan the observed (pre-impact) window and find the latest time index
    at which the required stopping distance (given the vehicle's speed at
    that instant) was still <= the remaining gap to the collision point.
    Returns (idx, time_s, margin_m) for the last such instant, or
    (None, None, None) if avoidance was never physically possible within
    the observed window.
    """
    best = None
    for i in range(len(t_s)):
        needed = required_stopping_distance(speed_mps[i], max_decel, reaction_time)
        margin = gap_to_collision_m[i] - needed
        if margin >= 0:
            best = (i, float(t_s[i]), float(margin))
    return best if best else (None, None, None)


def assess_avoidability(t_s, speed_mps, gap_to_collision_m, collision_time_s,
                         reconstructability_confidence="MEDIUM",
                         max_decel=DEFAULT_MAX_DECEL_MPS2, reaction_time=DEFAULT_REACTION_TIME_S,
                         min_confidence_for_verdict="MEDIUM",
                         lead_speed_mps=None, y_m=None):
    """
    Main entry point. t_s / speed_mps / gap_to_collision_m are aligned
    arrays for the striking vehicle over the observed pre-impact window.
    reconstructability_confidence should reflect the upstream geometry/
    scale/tracking confidence (see backend/reconstructability/score.py) --
    this is the uncertainty-inheritance the brief requires: if upstream
    confidence is too low, this module refuses to give a verdict at all.
    """
    assumptions = {"max_decel_mps2": max_decel, "reaction_time_s": reaction_time}
    evidence = []

    confidence_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    if confidence_rank.get(reconstructability_confidence, 0) < confidence_rank.get(min_confidence_for_verdict, 1):
        return CausalityResult(
            verdict="INSUFFICIENT_EVIDENCE",
            narrative=(
                "Insufficient evidence to determine avoidability: upstream speed/geometry "
                f"confidence ({reconstructability_confidence}) does not meet the threshold required "
                "for a physical avoidability assessment."
            ),
            assumptions=assumptions, confidence=reconstructability_confidence,
        )

    if len(t_s) < 4:
        return CausalityResult(
            verdict="INSUFFICIENT_EVIDENCE",
            narrative="Insufficient evidence to determine avoidability: too few pre-impact frames tracked.",
            assumptions=assumptions, confidence=reconstructability_confidence,
        )

    v0 = float(speed_mps[0])
    ttc0 = float(gap_to_collision_m[0] / v0) if v0 > 0.5 else None
    evidence.append(f"Earliest observed speed: {v0:.1f} m/s ({v0*3.6:.0f} km/h), "
                     f"gap to eventual collision point: {gap_to_collision_m[0]:.1f} m.")

    onset_idx, onset_time = detect_braking_onset(t_s, speed_mps)
    braking_detected = onset_idx is not None
    onset_before_impact = None
    if braking_detected:
        onset_before_impact = collision_time_s - onset_time
        evidence.append(f"Sustained deceleration detected beginning ~{onset_before_impact:.1f}s before impact.")
    else:
        evidence.append("No sustained deceleration (braking) detected in the observed window.")

    steering_detected = False
    steering_onset_before_impact = None
    steering_direction = None
    if y_m is not None and len(y_m) == len(t_s):
        steer_idx, steer_time, steer_dir = detect_steering_onset(t_s, y_m)
        steering_detected = steer_idx is not None
        if steering_detected:
            steering_onset_before_impact = collision_time_s - steer_time
            steering_direction = steer_dir
            evidence.append(
                f"Sustained lateral maneuver ({steer_dir}) detected beginning "
                f"~{steering_onset_before_impact:.1f}s before impact."
            )
        else:
            evidence.append("No sustained lateral maneuver (steering) detected in the observed window.")

    avoid_idx, avoid_time, margin_m = find_last_possible_avoidance_point(
        t_s, speed_mps, gap_to_collision_m, max_decel, reaction_time)

    if avoid_idx is None:
        verdict = "NOT_AVOIDABLE"
        avoidance_time_before_impact = None
        narrative = (
            f"Vehicle appears to have been traveling at approximately {v0:.1f} m/s "
            f"({v0*3.6:.0f} km/h) with a gap of {gap_to_collision_m[0]:.1f} m at the earliest "
            f"observed point. Under the modeled assumptions (max deceleration "
            f"{max_decel:.1f} m/s^2, reaction time {reaction_time:.1f}s), the required stopping "
            f"distance at every observed instant exceeded the available gap. "
            f"Based on reconstructed trajectories, the collision appears potentially "
            f"unavoidable under the modeled assumptions, even under immediate maximum braking "
            f"from the earliest observed point."
        )
        if steering_detected:
            narrative += (
                f" A sustained lateral maneuver ({steering_direction}) was also detected beginning "
                f"approximately {steering_onset_before_impact:.1f}s before impact."
            )
    else:
        verdict = "AVOIDABLE"
        avoidance_time_before_impact = collision_time_s - avoid_time
        narrative = (
            f"Vehicle appears to have had approximately {avoidance_time_before_impact:.1f}s of "
            f"potential avoidance time before the collision (last instant at which the modeled "
            f"stopping distance still fit within the observed gap, with a "
            f"{margin_m:.1f}m margin). "
        )
        if braking_detected:
            narrative += f"Evidence suggests braking began approximately {onset_before_impact:.1f}s before impact. "
        else:
            narrative += "No sustained braking was detected in the observed window. "
        if steering_detected:
            narrative += (
                f"A sustained lateral maneuver ({steering_direction}) was also detected beginning "
                f"approximately {steering_onset_before_impact:.1f}s before impact. "
            )
        narrative += (
            f"Based on reconstructed trajectories, the collision appears potentially avoidable "
            f"under the modeled assumptions (max deceleration {max_decel:.1f} m/s^2, reaction time "
            f"{reaction_time:.1f}s)."
        )

    lead_peak_decel = None
    if lead_speed_mps is not None and len(lead_speed_mps) == len(t_s) and len(t_s) >= 3:
        lead_accel = np.gradient(np.asarray(lead_speed_mps), t_s)
        min_accel = float(np.min(lead_accel))
        if min_accel < -0.5:  # only report if some real deceleration was observed
            lead_peak_decel = min_accel

    contribution_factors = compute_contribution_factors(
        v0, float(gap_to_collision_m[0]), lead_peak_decel, max_decel, reaction_time,
    )

    return CausalityResult(
        verdict=verdict,
        narrative=narrative,
        striking_vehicle_speed_at_earliest_observed_mps=v0,
        time_to_collision_at_earliest_observed_s=ttc0,
        braking_detected=braking_detected,
        braking_onset_time_before_impact_s=onset_before_impact,
        steering_detected=steering_detected,
        steering_onset_time_before_impact_s=steering_onset_before_impact,
        steering_direction=steering_direction,
        last_possible_avoidance_time_before_impact_s=avoidance_time_before_impact,
        assumptions=assumptions,
        confidence=reconstructability_confidence,
        evidence=evidence,
        contribution_factors=contribution_factors,
    )
