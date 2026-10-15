"""
The Skeptic: an adversarial second agent that cross-examines the
Reconstructor's (backend/agent/loop.py) accepted geometry/homography
before it's allowed to ship, using evidence the Reconstructor's own
acceptance test did NOT use to reach its conclusion.

Motivation is a real, documented near-miss (see docs/agent_tool_upgrades.md):
broadening the Reconstructor's roll-suspect trigger caused it to accept
a WRONG rotation angle that happened to coincidentally pass the
width-consistency + correspondence-span checks -- 0.03% speed error
became 91%, silently, under a confident RECONSTRUCTABLE verdict. The fix
at the time was to revert and stay conservative. This module is the
actual fix: rather than trusting the Reconstructor's own test more, or
making it stricter (which only rejects more real footage without
addressing WHY a wrong answer can pass it), give the pipeline a second,
independent check specifically designed to catch that failure mode, and
let it downgrade the verdict when it finds a real contradiction.

Two checks, deliberately NOT reusing the Reconstructor's own acceptance
signals (width-consistency max-error, correspondence span):

1. ANGLE AMBIGUITY (only when roll correction was applied): a
   well-identified correction should have one angle that clearly fits
   best. If another angle several degrees away ALSO scores near the
   acceptance threshold, the "winner" isn't actually unique -- exactly
   the shape of the reverted false positive (a loosened trigger let the
   search accept a coincidentally-passing wrong angle from a field of
   several similarly-scoring candidates).

2. DEPTH-TREND CHECK: the Reconstructor's width-consistency check uses a
   single max-error-magnitude scalar across sampled rows, which can hide
   a systematic bias (a residual, uncorrected rotation error makes
   apparent lane width drift linearly with depth, not scatter randomly)
   as long as no single row's error crosses the threshold. This fits a
   linear trend of per-row width error against depth and flags a real
   slope the scalar check can miss.

Either check finding a real problem downgrades confidence rather than
silently shipping a confident wrong answer -- and the whole exchange
(Reconstructor's evidence, Skeptic's objection or clearance) is logged
to the shared ToolLog trace, so the final report can show it as an
actual cross-examination, not a hidden internal flag.
"""
import numpy as np

from cv.homography.ground_plane import image_to_world

ANGLE_AMBIGUITY_MARGIN_DEG = 5.0       # candidate angles within this of the winner are "close", ignored
ANGLE_AMBIGUITY_SCORE_RATIO = 1.5      # a competing angle scoring within this multiple of the winner's error is a real rival
DEPTH_TREND_MIN_SLOPE_M_PER_100M = 0.6  # minimum width-error slope (m per 100m depth) worth flagging
DEPTH_TREND_MIN_R2 = 0.5                # only trust the trend if it explains a decent share of the variance


def _check_angle_ambiguity(roll_trace, best_angle_deg, best_width_error_m):
    """roll_trace: the `trace` list returned by
    cv.geometry.roll_correction.estimate_and_correct_roll -- entries
    with angle_deg / width_error_m / status. The Reconstructor never
    looks at this data itself once it has a winner; the Skeptic does."""
    if not roll_trace or best_angle_deg is None or best_width_error_m is None:
        return None  # roll correction wasn't used on this frame -- check doesn't apply
    rivals = [
        e for e in roll_trace
        if e.get("width_error_m") is not None
        and abs(e["angle_deg"] - best_angle_deg) > ANGLE_AMBIGUITY_MARGIN_DEG
        and e["width_error_m"] <= best_width_error_m * ANGLE_AMBIGUITY_SCORE_RATIO
    ]
    if not rivals:
        return {"disputed": False,
                "reason": f"winning angle {best_angle_deg:+d} deg has no close rival more than "
                          f"{ANGLE_AMBIGUITY_MARGIN_DEG:.0f} deg away -- correction looks well-identified."}
    rivals.sort(key=lambda e: e["width_error_m"])
    rival = rivals[0]
    return {
        "disputed": True,
        "reason": (f"angle {best_angle_deg:+d} deg was accepted (width error {best_width_error_m:.3f}m), but "
                   f"{rival['angle_deg']:+d} deg -- {abs(rival['angle_deg']-best_angle_deg):.0f} deg away -- "
                   f"scores nearly as well (width error {rival['width_error_m']:.3f}m). The correction is not "
                   f"uniquely identified; the accepted angle deserves reduced confidence."),
        "rival_angle_deg": rival["angle_deg"], "rival_width_error_m": rival["width_error_m"],
    }


def _check_depth_trend(H, left_line, right_line, image_h, assumed_width_m, n_samples=12, v_margin=0.15):
    """Independent of lane_width_self_consistency's own scalar summary
    (same sample rows, but a different statistic): fits per-row width
    error against depth and checks for a real linear trend, which a
    single max-error scalar can miss whenever no individual row crosses
    the threshold even though the errors aren't random."""
    ml, bl = left_line
    mr, br = right_line
    v_lo, v_hi = int(image_h * v_margin), int(image_h * (1 - v_margin))
    vs = np.linspace(v_lo, v_hi, n_samples)
    depths, errors = [], []
    for v in vs:
        ul, ur = ml * v + bl, mr * v + br
        world = image_to_world(H, [[ul, v], [ur, v]])
        depths.append(float((world[0, 0] + world[1, 0]) / 2.0))
        errors.append(float(abs(world[1, 1] - world[0, 1]) - assumed_width_m))
    depths, errors = np.array(depths), np.array(errors)
    if depths.max() - depths.min() < 1e-6:
        return {"disputed": False, "reason": "insufficient depth range sampled to test for a systematic trend."}

    slope, intercept = np.polyfit(depths, errors, 1)
    pred = slope * depths + intercept
    ss_res = float(np.sum((errors - pred) ** 2))
    ss_tot = float(np.sum((errors - errors.mean()) ** 2)) or 1e-9
    r2 = 1.0 - ss_res / ss_tot
    slope_per_100m = slope * 100.0

    disputed = bool(abs(slope_per_100m) >= DEPTH_TREND_MIN_SLOPE_M_PER_100M and r2 >= DEPTH_TREND_MIN_R2)
    reason = (f"lane-width error trends {slope_per_100m:+.2f}m per 100m of depth (R^2={r2:.2f}) -- a systematic "
              f"drift consistent with residual, uncorrected geometric bias, not random noise." if disputed else
              f"lane-width error vs. depth shows no significant systematic trend "
              f"(slope {slope_per_100m:+.2f}m/100m, R^2={r2:.2f}).")
    return {"disputed": disputed, "reason": reason, "slope_m_per_100m": slope_per_100m, "r2": r2}


def cross_examine(accepted, log=None):
    """
    accepted: {H, diag (with left_line/right_line/lane_width_self_consistency),
               image_h, assumed_lane_width_m, roll_correction_deg, roll_trace}
    Returns {disputed: bool, findings: [ {check, disputed, reason, ...}, ... ]}
    """
    findings = []

    diag = accepted.get("diag", {})
    angle_check = _check_angle_ambiguity(
        accepted.get("roll_trace"), accepted.get("roll_correction_deg"),
        (diag.get("lane_width_self_consistency") or {}).get("max_abs_error_vs_assumed_m"))
    if angle_check is not None:
        findings.append({"check": "angle_ambiguity", **angle_check})

    if diag.get("left_line") is not None and diag.get("right_line") is not None and accepted.get("H") is not None:
        depth_check = _check_depth_trend(
            accepted["H"], diag["left_line"], diag["right_line"],
            accepted["image_h"], accepted["assumed_lane_width_m"])
        findings.append({"check": "depth_trend", **depth_check})

    disputed = any(f["disputed"] for f in findings)
    if log is not None:
        for f in findings:
            log.record(f"skeptic_{f['check']}", {}, {"disputed": f["disputed"]}, reasoning=f["reason"])
        log.record("skeptic_verdict", {}, {"disputed": disputed},
                   reasoning=("The Skeptic found grounds to dispute the Reconstructor's accepted geometry -- "
                              "downgrading confidence rather than silently accepting it." if disputed else
                              "The Skeptic ran its independent checks and found no contradiction -- "
                              "the Reconstructor's answer stands, cross-examined."))

    return {"disputed": disputed, "findings": findings}
