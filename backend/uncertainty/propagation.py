"""
Uncertainty propagation (brief section 15).

Nothing in the pipeline before this module ever turns its diagnostics
into a stated confidence interval on the final speed number -- every
speed/verdict is currently a point estimate, even though the brief
explicitly calls this out as one of Stradiva's core differentiators and
warns against "fake precision" (42.17 mph vs 38-46 mph).

Design choice, stated up front: this is an ENGINEERING HEURISTIC, not a
rigorous statistical error propagation (that would need a measured
noise/variance model for every upstream CV stage -- line detection,
homography fitting, tracking -- which doesn't exist here and would be
its own multi-week project). What it *does* do honestly:

  1. Turn diagnostics the pipeline already computes (lane-width
     self-consistency error, homography inlier fraction, tracking loss
     rate, frame rate, roll-correction residual) into per-source
     relative-uncertainty contributions using simple, documented,
     monotonic mappings -- more inlier noise/tracking loss/roll residual
     always means a wider stated interval, never a narrower one.
  2. Combine sources in quadrature (assumes they're roughly independent,
     the standard first-order assumption when no covariance structure is
     known) into one relative uncertainty.
  3. Scale that combined figure by a single calibration constant
     (CI90_SCALE) so the stated interval is an actual ~90% interval in
     practice, not just a plausible-looking one -- CI90_SCALE is fit by
     `evaluation/scripts/run_uncertainty_calibration.py`, which measures
     empirical coverage (how often ground-truth speed actually falls
     inside the stated interval) across many synthetic trials and
     reports that number. If coverage drifts from ~90% as the dataset
     grows, refit the constant there -- don't hand-tune it here.

This module never claims a source of uncertainty it hasn't actually
measured, and never narrows the interval based on anything not present
in the diagnostics dict.
"""
from dataclasses import dataclass, field

ASSUMED_LANE_WIDTH_M = 3.7  # matches ASSUMED_LANE_HALF_WIDTH_M * 2 everywhere else in the codebase

# Fit by evaluation/scripts/run_uncertainty_calibration.py -- see
# docs/uncertainty_calibration.md for the coverage measurement behind
# this number (28 synthetic trials across 4 speeds x 7 roll angles, 19
# reconstructable after adding the agent's neighbor-frame search;
# empirical 90% CI coverage at candidate scales: 1.0->68.4%,
# 1.25->73.7%, 1.5->78.9%, 2.0->84.2%, 2.5/3.0->94.7%. 2.5 is the
# closest grid point to the 90% target and is used here; refit as the
# dataset grows rather than hand-tuning further from one sweep).
CI90_SCALE = 2.5

MAX_RELATIVE_UNCERTAINTY = 0.95  # never state a CI that swallows the point estimate entirely


@dataclass
class UncertaintyResult:
    relative_uncertainty: float  # fraction, pre-CI90_SCALE, "1-sigma-like"
    sources: dict = field(default_factory=dict)  # name -> relative contribution
    dominant_sources: list = field(default_factory=list)  # names, largest contribution first
    ci90_half_width_frac: float = 0.0

    def interval(self, point_estimate):
        half_width = point_estimate * self.ci90_half_width_frac
        return max(0.0, point_estimate - half_width), point_estimate + half_width


def _scale_source(width_consistency):
    """Relative scale/calibration error, directly from the lane-width
    self-consistency check: the max reprojection error vs. the assumed
    lane width IS the fractional error in the ground-plane's metric
    scale, since that assumed width is the only physical ruler the
    whole homography is calibrated against."""
    if not width_consistency:
        return 0.35  # width check unavailable -- treat as high uncertainty, not zero
    err_m = width_consistency.get("max_abs_error_vs_assumed_m")
    if err_m is None:
        return 0.35
    return min(1.0, err_m / ASSUMED_LANE_WIDTH_M)


def _inlier_source(homography_inlier_fraction):
    """Lower RANSAC inlier fraction -> noisier correspondence set behind
    the homography, independent of whether the mean fit still passed the
    width-consistency check."""
    if homography_inlier_fraction is None:
        return 0.20
    return max(0.0, (1.0 - homography_inlier_fraction)) * 0.5


def _geometry_source(n_left_segments, n_right_segments, vp_stability_px_std):
    """Vanishing-point / edge-fit stability. Uses the direct px-std
    signal when available; falls back to segment count (fewer edge
    segments -> noisier line fit) since vp_stability_px_std is often
    None in this codebase's diagnostics (see reconstructability/score.py)."""
    if vp_stability_px_std is not None:
        return min(1.0, vp_stability_px_std / 30.0) * 0.25
    n_min = min(n_left_segments or 0, n_right_segments or 0)
    return max(0.0, (8 - n_min) / 8.0) * 0.20


def _tracking_source(n_tracked, n_lost):
    total = n_tracked + n_lost
    if total == 0:
        return 0.5
    lost_frac = n_lost / total
    return lost_frac * 0.4


def _temporal_source(fps):
    if not fps or fps <= 0:
        return 0.15
    return max(0.0, min(1.0, (20.0 - fps) / 20.0)) * 0.15


def _roll_residual_source(roll_correction_deg):
    """De-rotation is itself imperfect (rotation about the image center
    rather than the true, generally-unknown principal point; a coarse
    angle-search grid). Larger corrections carry more unmodeled residual
    than a naturally level camera."""
    if not roll_correction_deg:
        return 0.0
    return min(1.0, abs(roll_correction_deg) / 45.0) * 0.10


def propagate_speed_uncertainty(geometry_diag, tracking_info, fps,
                                 roll_correction_deg=0.0, ci90_scale=CI90_SCALE):
    """
    geometry_diag: the pipeline's geometry diagnostics dict (same shape as
      result["geometry"] -- must include lane_width_self_consistency and
      homography_inlier_fraction for a non-degenerate estimate).
    tracking_info: dict with n_tracked / n_lost for the vehicle whose
      speed is being reported.
    fps: source frame rate.
    roll_correction_deg: geometry_diag.get("roll_correction_applied_deg"),
      passed explicitly so callers can't forget it.
    """
    sources = {
        "scale_calibration": _scale_source(geometry_diag.get("lane_width_self_consistency")),
        "homography_inliers": _inlier_source(geometry_diag.get("homography_inlier_fraction")),
        "geometry_stability": _geometry_source(
            geometry_diag.get("n_left_segments"), geometry_diag.get("n_right_segments"),
            geometry_diag.get("vp_stability_px_std")),
        "tracking_loss": _tracking_source(
            tracking_info.get("n_tracked", 0), tracking_info.get("n_lost", 0)),
        "frame_rate": _temporal_source(fps),
        "roll_correction_residual": _roll_residual_source(roll_correction_deg),
    }
    combined = min(MAX_RELATIVE_UNCERTAINTY, sum(v ** 2 for v in sources.values()) ** 0.5)
    dominant = sorted(sources, key=sources.get, reverse=True)
    dominant = [name for name in dominant if sources[name] > 0.01]

    return UncertaintyResult(
        relative_uncertainty=round(combined, 4),
        sources={k: round(v, 4) for k, v in sources.items()},
        dominant_sources=dominant[:3],
        ci90_half_width_frac=round(min(MAX_RELATIVE_UNCERTAINTY, combined * ci90_scale), 4),
    )
