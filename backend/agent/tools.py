"""
Agent tool interface (brief section 18): explicit, individually callable
wrappers around the CV/reconstruction stages, each returning a
structured result plus enough diagnostic detail for the controller
(backend/agent/loop.py) to decide what to do next. The agent orchestrates
these; it never invents a measurement itself -- every number returned
here comes from an actual OpenCV/numpy computation on real pixels.

Each tool call is logged (name, args summary, result summary) so the
full trace is inspectable -- required for the observability/logging
item in section 17/19 and for the "reproducible agentic trace"
deliverable.
"""
import time

from cv.lines.line_detection import detect_line_segments, cluster_by_slope_side, fit_line_x_of_y
from cv.homography.ground_plane import (
    detect_dash_rows, build_correspondences, fit_ground_homography,
    lane_width_self_consistency,
)


def analyze_frame_geometry(frame, w, h, lane_half_width_m, dash_len_m, dash_gap_m):
    """Tool: fit lane-edge lines + vanishing-point-implicit geometry from ONE frame."""
    segs, _ = detect_line_segments(frame)
    left_segs, right_segs = cluster_by_slope_side(segs, w)
    left_line = fit_line_x_of_y(left_segs)
    right_line = fit_line_x_of_y(right_segs)
    diag = {
        "n_left_segments": int(len(left_segs)), "n_right_segments": int(len(right_segs)),
        "left_line": left_line, "right_line": right_line,
    }
    return diag


def estimate_ground_plane(frame, left_line, right_line, w, h,
                           lane_half_width_m, dash_len_m, dash_gap_m):
    """Tool: dash detection + scale-anchored homography fit, given edge lines."""
    if left_line is None or right_line is None:
        return None, {"n_correspondence_points": 0, "homography_inlier_fraction": None}
    dashes = detect_dash_rows(frame, left_line, right_line)
    img_pts, world_pts = build_correspondences(dashes, left_line, right_line,
                                                lane_half_width_m, dash_len_m, dash_gap_m)
    diag = {"n_dashes_detected": len(dashes), "n_correspondence_points": int(len(img_pts))}
    if len(img_pts) < 4:
        return None, diag
    # world-X span covered by the correspondence points: a homography
    # fit only on a narrow forward-distance range extrapolates badly to
    # positions far outside that range (found empirically: a fit
    # calibrated over 0-15m produced a near-perfect lateral-width check
    # but was wildly wrong applied to vehicle positions out at 60-100m --
    # width-consistency alone doesn't catch this, since it only tests
    # points within the same narrow row/depth range the fit already saw).
    diag["world_x_span_m"] = float(world_pts[:, 0].max() - world_pts[:, 0].min()) if len(world_pts) else 0.0
    H, mask = fit_ground_homography(img_pts, world_pts)
    diag["homography_inlier_fraction"] = float(mask.mean()) if mask is not None else None
    if H is not None:
        diag["lane_width_self_consistency"] = lane_width_self_consistency(
            H, left_line, right_line, h, lane_half_width_m * 2)
    return H, diag


class ToolLog:
    """Records every agent tool call for the reproducible trace."""
    def __init__(self):
        self.calls = []

    def record(self, tool_name, args_summary, result_summary, reasoning=""):
        self.calls.append({
            "t": time.time(), "tool": tool_name, "args": args_summary,
            "result": result_summary, "reasoning": reasoning,
        })

    def as_list(self):
        return list(self.calls)
