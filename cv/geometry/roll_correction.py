"""
Camera roll correction.

Every geometry/scale module in this codebase relies on one fact that's
only exactly true for a level (zero-roll) camera: image row v depends
only on world forward-distance X, not lateral offset Y (see
cv/homography/ground_plane.py's module docstring). Week 2 found that a
rolled camera silently breaks this and produces a large, real speed
error (45%) while other components still scored well -- fixed then by
adding a hard veto on the lane-width self-consistency check. This module
goes one step further: instead of only detecting roll and refusing,
actively estimate and correct for it so genuinely rolled footage (very
common for handheld/dashcam sources) doesn't have to be rejected outright.

Approach: camera roll is, to first order, equivalent to a 2D rotation of
the captured image about the principal point (this falls directly out of
the pinhole model -- an in-plane rotation of the camera's right/down
axes rotates image-plane coordinates by the same angle, independent of
depth). So "correcting" roll means finding the rotation angle that,
when undone, makes the already-existing lane-width self-consistency
check happy again -- i.e. search over candidate angles, and use the
SAME diagnostic Week 2 built to detect the problem as the objective
function for fixing it. This sidesteps ever needing to derive the
correction angle's sign/convention analytically: whichever candidate
empirically minimizes the width-consistency error is used, regardless
of the direction convention.
"""
import cv2
import numpy as np

from cv.lines.line_detection import detect_line_segments, cluster_by_slope_side, fit_line_x_of_y
from cv.homography.ground_plane import (
    detect_dash_rows, build_correspondences, fit_ground_homography, lane_width_self_consistency,
)

DEFAULT_ANGLE_RANGE_DEG = list(range(-15, 16, 3))  # coarse pass; refined below


def derotate_frame(frame, angle_deg, center=None):
    """Rotate the frame by angle_deg about its center (or a given
    principal point) to undo an assumed camera roll of that angle."""
    h, w = frame.shape[:2]
    center = center or (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _fit_geometry_and_score(frame, w, h, lane_half_width_m, dash_len_m, dash_gap_m, min_span_m=30.0):
    """Runs the full geometry->scale chain on one (possibly derotated)
    frame and returns (H, diag, width_consistency_error) or (None, diag, None).
    Rejects fits calibrated over too narrow a depth range (see
    backend/agent/tools.py's world_x_span_m for why) by returning no
    usable score for them, same acceptance bar as the main agent loop."""
    segs, _ = detect_line_segments(frame)
    left_segs, right_segs = cluster_by_slope_side(segs, w)
    left_line = fit_line_x_of_y(left_segs)
    right_line = fit_line_x_of_y(right_segs)
    if left_line is None or right_line is None:
        return None, {"status": "no lane edges"}, None

    dashes = detect_dash_rows(frame, left_line, right_line)
    img_pts, world_pts = build_correspondences(dashes, left_line, right_line,
                                                lane_half_width_m, dash_len_m, dash_gap_m)
    if len(img_pts) < 4:
        return None, {"status": "insufficient correspondence points"}, None
    span_m = float(world_pts[:, 0].max() - world_pts[:, 0].min())
    if span_m < min_span_m:
        return None, {"status": f"correspondence span only {span_m:.0f}m (need >= {min_span_m:.0f}m)"}, None

    H, mask = fit_ground_homography(img_pts, world_pts)
    if H is None:
        return None, {"status": "homography fit failed"}, None

    width_check = lane_width_self_consistency(H, left_line, right_line, h, lane_half_width_m * 2)
    diag = {
        "left_line": left_line, "right_line": right_line,
        "n_correspondence_points": int(len(img_pts)), "world_x_span_m": span_m,
        "homography_inlier_fraction": float(mask.mean()) if mask is not None else None,
        "lane_width_self_consistency": width_check,
    }
    return H, diag, width_check["max_abs_error_vs_assumed_m"]


def estimate_and_correct_roll(frame, w, h, lane_half_width_m, dash_len_m, dash_gap_m,
                               coarse_angles=None, refine_window=4, refine_step=1, log=None):
    """
    Coarse-to-fine search over candidate roll angles, scored by the
    lane-width self-consistency error at each. Returns a dict:
      {best_angle_deg, H, diag, width_error, corrected_frame, trace}
    or None if no candidate improved on the uncorrected frame.
    """
    coarse_angles = coarse_angles or DEFAULT_ANGLE_RANGE_DEG
    trace = []

    def evaluate(angle_deg):
        test_frame = frame if angle_deg == 0 else derotate_frame(frame, angle_deg)
        H, diag, err = _fit_geometry_and_score(test_frame, w, h, lane_half_width_m, dash_len_m, dash_gap_m)
        trace.append({"angle_deg": angle_deg, "width_error_m": err, "status": diag.get("status", "ok")})
        if log is not None:
            log.record("estimate_and_correct_camera_roll", {"angle_deg": angle_deg},
                       {"width_error_m": err, "status": diag.get("status", "ok")},
                       reasoning=f"Testing roll-correction hypothesis at {angle_deg:+d} deg.")
        return H, diag, err, test_frame

    results = [evaluate(a) for a in coarse_angles]
    valid = [(a, H, diag, err, tf) for a, (H, diag, err, tf) in zip(coarse_angles, results) if err is not None]
    if not valid:
        return None
    valid.sort(key=lambda x: x[3])
    best_coarse_angle = valid[0][0]

    fine_angles = [best_coarse_angle + d for d in
                   range(-refine_window, refine_window + 1, refine_step) if d != 0]
    fine_results = [evaluate(a) for a in fine_angles]
    all_candidates = valid + [(a, H, diag, err, tf) for a, (H, diag, err, tf)
                               in zip(fine_angles, fine_results) if err is not None]
    all_candidates.sort(key=lambda x: x[3])
    best_angle, best_H, best_diag, best_err, best_frame = all_candidates[0]

    return {
        "best_angle_deg": best_angle, "H": best_H, "diag": best_diag,
        "width_error_m": best_err, "corrected_frame": best_frame, "trace": trace,
    }
