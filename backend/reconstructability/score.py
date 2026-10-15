"""
Stage: reconstructability assessment (brief section 16).

This is a core product feature, not an error path: every video should
get a score + verdict explaining *why*, with absolute-position
confidence reported separately from relative-motion/speed confidence
(the dash-anchoring failure mode from Week 1 makes those two genuinely
different claims -- see docs/week1_feasibility.md).

Component scores are 0-100. The overall score is a weighted combination,
but the verdict is also gated by hard floors: a catastrophic single
component (e.g. no lane geometry at all) should veto reconstruction even
if it wouldn't drag the weighted average below threshold on its own.
"""
from dataclasses import dataclass, field


WEIGHTS = {
    "geometry": 0.30,
    "scale": 0.30,
    "tracking": 0.25,
    "temporal": 0.15,
}
HARD_FLOORS = {  # any component below its floor vetoes reconstruction regardless of weighted score
    "geometry": 20,
    "scale": 15,
    "tracking": 30,
}
OVERALL_PASS_THRESHOLD = 55

# A severe lane-width self-consistency failure is treated as an independent
# veto, not just a drag on the blended scale score. Rationale (found the
# hard way in Week 2 stress testing): a single violated assumption -- e.g.
# camera roll -- can produce a large real speed error (45% in the roll
# stress test) while still scoring "fine" on a blended average, because
# geometry/tracking/temporal stayed high and diluted the one component
# that actually caught the problem. A hard veto on the direct diagnostic
# closes that hole; a blended score alone did not.
WIDTH_CONSISTENCY_VETO_ERROR_M = 1.0


@dataclass
class ReconstructabilityResult:
    overall: float
    components: dict
    absolute_position_confidence: str  # "LOW" | "MEDIUM" | "HIGH"
    verdict: str
    reasons: list = field(default_factory=list)


def score_geometry(n_left_segments, n_right_segments, vp_stability_px_std):
    if n_left_segments < 2 or n_right_segments < 2:
        return 0.0, ["fewer than 2 line segments on one or both lane edges"]
    reasons = []
    seg_score = min(100, 15 * (n_left_segments + n_right_segments))
    if vp_stability_px_std is None:
        stability_score = 30
        reasons.append("vanishing point stability could not be estimated")
    else:
        # 0px std -> 100, 30px+ std -> 0
        stability_score = max(0, 100 - (vp_stability_px_std / 30.0) * 100)
        if vp_stability_px_std > 15:
            reasons.append(f"vanishing point unstable (±{vp_stability_px_std:.1f}px)")
    return 0.5 * seg_score + 0.5 * stability_score, reasons


def score_scale(n_correspondence_points, homography_inlier_fraction, width_consistency):
    if n_correspondence_points < 8 or homography_inlier_fraction is None:
        return 0.0, ["insufficient correspondence points for scale recovery"]
    reasons = []
    density_score = min(100, n_correspondence_points * 3)
    inlier_score = homography_inlier_fraction * 100
    if homography_inlier_fraction < 0.5:
        reasons.append(f"low homography inlier fraction ({homography_inlier_fraction:.0%})")
    consistency_score = 100 if width_consistency and width_consistency.get("consistent") else 30
    if width_consistency and not width_consistency.get("consistent"):
        reasons.append(
            f"lane-width self-consistency check failed "
            f"(std={width_consistency['std_width_m']:.2f}m, "
            f"err={width_consistency['max_abs_error_vs_assumed_m']:.2f}m vs assumed)"
        )
    return 0.4 * density_score + 0.35 * inlier_score + 0.25 * consistency_score, reasons


def score_tracking(n_tracked, n_lost):
    total = n_tracked + n_lost
    if total == 0:
        return 0.0, ["no frames processed"]
    frac = n_tracked / total
    reasons = [] if frac > 0.9 else [f"vehicle lost on {n_lost}/{total} frames"]
    return frac * 100, reasons


def score_temporal(fps, n_frames):
    reasons = []
    if fps <= 0 or n_frames < 5:
        return 0.0, ["insufficient frames/fps for motion estimation"]
    # below 10fps, velocity estimates get unreliable; above ~24fps, diminishing returns
    fps_score = min(100, max(0, (fps - 5) / (24 - 5) * 100))
    if fps < 15:
        reasons.append(f"low frame rate ({fps:.1f} fps) increases velocity-estimate noise")
    duration_s = n_frames / fps
    duration_score = min(100, duration_s / 2.0 * 100)  # want >=2s of motion to smooth over
    if duration_s < 1.0:
        reasons.append(f"short observation window ({duration_s:.1f}s)")
    return 0.7 * fps_score + 0.3 * duration_score, reasons


def assess_reconstructability(geometry_diag, scores_input, width_consistency=None):
    """
    geometry_diag: dict from stage2_geometry diagnostics
    scores_input: dict with n_tracked, n_lost, fps, n_frames
    """
    reasons_all = []

    g_score, g_reasons = score_geometry(
        geometry_diag.get("n_left_segments", 0),
        geometry_diag.get("n_right_segments", 0),
        geometry_diag.get("vp_stability_px_std"),
    )
    s_score, s_reasons = score_scale(
        geometry_diag.get("n_correspondence_points", 0),
        geometry_diag.get("homography_inlier_fraction"),
        width_consistency,
    )
    t_score, t_reasons = score_tracking(scores_input.get("n_tracked", 0), scores_input.get("n_lost", 0))
    temp_score, temp_reasons = score_temporal(scores_input.get("fps", 0), scores_input.get("n_frames", 0))

    reasons_all += g_reasons + s_reasons + t_reasons + temp_reasons

    components = {
        "geometry": round(g_score, 1),
        "scale": round(s_score, 1),
        "tracking": round(t_score, 1),
        "temporal": round(temp_score, 1),
    }
    overall = sum(WEIGHTS[k] * v for k, v in components.items())

    vetoed = any(components[k] < floor for k, floor in HARD_FLOORS.items())
    if width_consistency and not width_consistency.get("consistent", True):
        width_err = width_consistency.get("max_abs_error_vs_assumed_m", 0.0)
        if width_err > WIDTH_CONSISTENCY_VETO_ERROR_M:
            vetoed = True
            reasons_all.append(
                f"VETO: lane-width self-consistency error ({width_err:.2f}m) exceeds "
                f"{WIDTH_CONSISTENCY_VETO_ERROR_M}m -- ground-plane mapping is not trustworthy "
                f"even though other components scored well"
            )
    passed = overall >= OVERALL_PASS_THRESHOLD and not vetoed

    # absolute position confidence is orthogonal to the above: it depends
    # on whether an independently-verified spatial anchor was supplied.
    # None wired in yet (Week 1/2 scope) -> always LOW for now, by design.
    absolute_position_confidence = "LOW"
    if absolute_position_confidence == "LOW":
        reasons_all.append(
            "absolute position is unanchored (no independently-verified landmark); "
            "treat absolute displacement as low-confidence, relative motion/speed as the reliable output"
        )

    if vetoed:
        verdict = "NOT_RECONSTRUCTABLE"
    elif passed:
        verdict = "RECONSTRUCTABLE"
    else:
        verdict = "RECONSTRUCTABLE_LOW_CONFIDENCE"

    return ReconstructabilityResult(
        overall=round(overall, 1),
        components=components,
        absolute_position_confidence=absolute_position_confidence,
        verdict=verdict,
        reasons=reasons_all,
    )
