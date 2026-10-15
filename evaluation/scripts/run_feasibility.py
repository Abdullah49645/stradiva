"""
Week-1 feasibility run (brief section 38): take one controlled video,
recover a plausible physical trajectory, and compare against
independently-known ground truth. No UI, no AWS -- just: does the core
reconstruction math work.

Pipeline: video ingestion -> line detection -> edge-line fit ->
vanishing point (diagnostic) -> dash detection -> scale-anchored
ground-plane homography -> per-frame vehicle ground-contact tracking
-> image-to-world projection -> smoothing -> velocity estimate ->
compare to ground truth.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cv.lines.line_detection import detect_line_segments, cluster_by_slope_side, fit_line_x_of_y
from cv.vanishing_points.vp_estimation import intersect_x_of_y_lines, estimate_vp_stability
from cv.homography.ground_plane import (
    detect_dash_rows, build_correspondences, fit_ground_homography, image_to_world,
    lane_width_self_consistency,
)
from cv.tracking.vehicle_tracker import track_ground_contact_point
from backend.reconstruction.trajectory import (
    smooth_trajectory, velocity_from_position, relative_displacement,
)
from backend.reconstructability.score import assess_reconstructability
from backend.agent.loop import run_geometry_agent
from cv.geometry.roll_correction import derotate_frame

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
VIDEO_PATH = os.path.join(DATASET_DIR, "synthetic_scene_A.mp4")
GT_PATH = os.path.join(DATASET_DIR, "synthetic_scene_A_ground_truth.json")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
os.makedirs(REPORT_DIR, exist_ok=True)

# priors used only for scale recovery (flagged explicitly, per section 12 --
# the pipeline must know and report *where* its scale assumption came from)
ASSUMED_LANE_HALF_WIDTH_M = 1.85
ASSUMED_DASH_LEN_M = 3.0
ASSUMED_DASH_GAP_M = 9.0


def load_ground_truth():
    with open(GT_PATH) as f:
        return json.load(f)


def stage1_ingest(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return {"fps": fps, "n_frames": n, "w": w, "h": h, "frames_read": len(frames)}, frames


def stage2_geometry(frames, w, h):
    """Fit lane edges + vanishing point + dash-based ground homography."""
    # use a mid-sequence frame where the vehicle is far away (small silhouette,
    # minimal occlusion of the road markings)
    ref_frame = frames[3]
    segs, edges = detect_line_segments(ref_frame)
    left_segs, right_segs = cluster_by_slope_side(segs, w)
    left_line = fit_line_x_of_y(left_segs)
    right_line = fit_line_x_of_y(right_segs)

    diagnostics = {
        "n_segments_detected": int(len(segs)),
        "n_left_segments": int(len(left_segs)),
        "n_right_segments": int(len(right_segs)),
        "left_line_fit": left_line,
        "right_line_fit": right_line,
    }

    if left_line is None or right_line is None:
        diagnostics["status"] = "FAILED: could not fit both lane edges"
        return None, diagnostics

    vp = intersect_x_of_y_lines(left_line, right_line)
    vp_mean, vp_std = estimate_vp_stability(left_segs, right_segs, fit_line_x_of_y)
    diagnostics["vanishing_point"] = vp
    diagnostics["vp_stability_px_std"] = float(vp_std) if vp_std is not None else None

    dashes = detect_dash_rows(ref_frame, left_line, right_line)
    diagnostics["n_dashes_detected"] = len(dashes)
    img_pts, world_pts = build_correspondences(
        dashes, left_line, right_line,
        ASSUMED_LANE_HALF_WIDTH_M, ASSUMED_DASH_LEN_M, ASSUMED_DASH_GAP_M,
    )
    diagnostics["n_correspondence_points"] = int(len(img_pts))

    H, mask = fit_ground_homography(img_pts, world_pts)
    if H is None:
        diagnostics["status"] = "FAILED: homography fit failed (insufficient dash correspondences)"
        return None, diagnostics

    inlier_frac = float(mask.mean()) if mask is not None else None
    diagnostics["homography_inlier_fraction"] = inlier_frac
    width_check = lane_width_self_consistency(H, left_line, right_line, h, ASSUMED_LANE_HALF_WIDTH_M * 2)
    diagnostics["lane_width_self_consistency"] = width_check
    diagnostics["status"] = "OK"
    return H, diagnostics


def stage3_track_and_reconstruct(frames, H, fps):
    prev_pt = None
    t_s, u_list, v_list = [], [], []
    n_lost = 0
    for i, frame in enumerate(frames):
        pt, conf = track_ground_contact_point(frame, prev_pt)
        if pt is None:
            n_lost += 1
            continue
        prev_pt = pt
        t_s.append(i / fps)
        u_list.append(pt[0])
        v_list.append(pt[1])

    uv = np.stack([u_list, v_list], axis=1)
    world = image_to_world(H, uv)
    x_raw, y_raw = world[:, 0], world[:, 1]
    t_s = np.array(t_s)

    x_smooth, y_smooth = smooth_trajectory(t_s, x_raw, y_raw)
    vx, vy, speed = velocity_from_position(t_s, x_smooth, y_smooth)

    return {
        "t_s": t_s, "x_raw_m": x_raw, "y_raw_m": y_raw,
        "x_smooth_m": x_smooth, "y_smooth_m": y_smooth,
        "speed_mps": speed, "n_tracked": len(t_s), "n_lost": n_lost,
    }


def score_against_ground_truth(recon, gt):
    gt_frames = gt["frames"]
    gt_t = np.array([f["t_s"] for f in gt_frames])
    gt_x = np.array([f["world_x_m"] for f in gt_frames])
    gt_speed_true = gt["vehicle"]["speed_true_mps"]

    x_interp = np.interp(recon["t_s"], gt_t, gt_x)
    pos_err = recon["x_smooth_m"] - x_interp
    pos_mae = float(np.mean(np.abs(pos_err)))
    pos_rmse = float(np.sqrt(np.mean(pos_err ** 2)))

    # relative displacement (anchor-independent) -- should be far more accurate
    # than absolute position; see backend/reconstruction/trajectory.py
    dx_recon, _ = relative_displacement(recon["x_smooth_m"], recon["y_smooth_m"])
    gt_x_interp_full = np.interp(recon["t_s"], gt_t, gt_x)
    dx_gt = gt_x_interp_full - gt_x_interp_full[0]
    disp_err = dx_recon - dx_gt
    disp_mae = float(np.mean(np.abs(disp_err)))
    disp_rmse = float(np.sqrt(np.mean(disp_err ** 2)))

    # drop edge frames (derivative artifacts at sequence boundaries)
    core = slice(len(recon["speed_mps"]) // 6, -max(1, len(recon["speed_mps"]) // 6))
    speed_core = recon["speed_mps"][core]
    speed_err = speed_core - gt_speed_true
    speed_mae = float(np.mean(np.abs(speed_err)))
    speed_rmse = float(np.sqrt(np.mean(speed_err ** 2)))
    speed_mean_est = float(np.mean(speed_core))

    return {
        "position_mae_m": pos_mae,
        "position_rmse_m": pos_rmse,
        "displacement_mae_m": disp_mae,
        "displacement_rmse_m": disp_rmse,
        "speed_true_mps": gt_speed_true,
        "speed_estimated_mean_mps": speed_mean_est,
        "speed_mae_mps": speed_mae,
        "speed_rmse_mps": speed_rmse,
        "speed_error_pct": float(100 * abs(speed_mean_est - gt_speed_true) / gt_speed_true),
    }


def run_pipeline(video_path=VIDEO_PATH, gt_path=GT_PATH, save_plot=True):
    """Reusable entry point (used by run_feasibility.py's main() and by
    run_stress_tests.py against other synthetic scenarios). Uses the
    agentic geometry search (including roll correction) rather than a
    single fixed reference frame, matching the two-vehicle collision
    pipeline (evaluation/scripts/run_causality_feasibility.py)."""
    with open(gt_path) as f:
        gt = json.load(f)
    ingest_info, frames = stage1_ingest(video_path)
    agent_result = run_geometry_agent(frames, ingest_info["w"], ingest_info["h"],
                                       ASSUMED_LANE_HALF_WIDTH_M, ASSUMED_DASH_LEN_M, ASSUMED_DASH_GAP_M)
    H, geo_diag = agent_result.H, agent_result.diag
    result = {"ingest": ingest_info, "geometry": geo_diag, "agent_trace": {
        "success": agent_result.success, "chosen_frame_idx": agent_result.chosen_frame_idx,
        "n_iterations": agent_result.n_iterations, "stop_reason": agent_result.stop_reason,
        "roll_correction_deg": agent_result.roll_correction_deg,
    }}
    if agent_result.roll_correction_deg:
        frames = [derotate_frame(f, agent_result.roll_correction_deg) for f in frames]
        result["geometry"]["roll_correction_applied_deg"] = agent_result.roll_correction_deg

    if H is None:
        assessment = assess_reconstructability(
            geo_diag, {"n_tracked": 0, "n_lost": 0, "fps": ingest_info["fps"], "n_frames": ingest_info["n_frames"]},
        )
        result["reconstructability"] = {
            "overall": assessment.overall, "components": assessment.components,
            "absolute_position_confidence": assessment.absolute_position_confidence,
            "verdict": assessment.verdict, "reasons": assessment.reasons,
        }
        result["status"] = "NOT_RECONSTRUCTABLE"
        return result

    recon = stage3_track_and_reconstruct(frames, H, ingest_info["fps"])
    scores = score_against_ground_truth(recon, gt)
    result["tracking"] = {"n_tracked": recon["n_tracked"], "n_lost": recon["n_lost"]}
    result["scores"] = scores

    assessment = assess_reconstructability(
        geo_diag,
        {"n_tracked": recon["n_tracked"], "n_lost": recon["n_lost"],
         "fps": ingest_info["fps"], "n_frames": ingest_info["n_frames"]},
        width_consistency=geo_diag.get("lane_width_self_consistency"),
    )
    result["reconstructability"] = {
        "overall": assessment.overall, "components": assessment.components,
        "absolute_position_confidence": assessment.absolute_position_confidence,
        "verdict": assessment.verdict, "reasons": assessment.reasons,
    }
    result["status"] = assessment.verdict
    return result


def main():
    gt = load_ground_truth()
    ingest_info, frames = stage1_ingest(VIDEO_PATH)
    H, geo_diag = stage2_geometry(frames, ingest_info["w"], ingest_info["h"])

    result = {"ingest": ingest_info, "geometry": geo_diag}

    if H is None:
        assessment = assess_reconstructability(
            geo_diag, {"n_tracked": 0, "n_lost": 0, "fps": ingest_info["fps"], "n_frames": ingest_info["n_frames"]},
        )
        result["reconstructability"] = {
            "overall": assessment.overall, "components": assessment.components,
            "absolute_position_confidence": assessment.absolute_position_confidence,
            "verdict": assessment.verdict, "reasons": assessment.reasons,
        }
        result["status"] = "NOT_RECONSTRUCTABLE"
        print(json.dumps(result, indent=2, default=str))
        with open(os.path.join(REPORT_DIR, "week1_run_result.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
        return

    recon = stage3_track_and_reconstruct(frames, H, ingest_info["fps"])
    scores = score_against_ground_truth(recon, gt)
    result["tracking"] = {"n_tracked": recon["n_tracked"], "n_lost": recon["n_lost"]}
    result["scores"] = scores

    assessment = assess_reconstructability(
        geo_diag,
        {"n_tracked": recon["n_tracked"], "n_lost": recon["n_lost"],
         "fps": ingest_info["fps"], "n_frames": ingest_info["n_frames"]},
        width_consistency=geo_diag.get("lane_width_self_consistency"),
    )
    result["reconstructability"] = {
        "overall": assessment.overall,
        "components": assessment.components,
        "absolute_position_confidence": assessment.absolute_position_confidence,
        "verdict": assessment.verdict,
        "reasons": assessment.reasons,
    }
    result["status"] = assessment.verdict

    with open(os.path.join(REPORT_DIR, "week1_run_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    # plot: reconstructed vs ground-truth trajectory + speed
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_t = np.array([f["t_s"] for f in gt["frames"]])
    gt_x = np.array([f["world_x_m"] for f in gt["frames"]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(gt_t, gt_x, label="ground truth", lw=2)
    axes[0].plot(recon["t_s"], recon["x_raw_m"], ".", ms=3, alpha=0.4, label="raw tracked")
    axes[0].plot(recon["t_s"], recon["x_smooth_m"], label="smoothed reconstruction", lw=2)
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("world X (m)")
    axes[0].set_title("Down-road position"); axes[0].legend(fontsize=8)

    axes[1].axhline(gt["vehicle"]["speed_true_mps"], color="k", ls="--", label="ground truth speed")
    axes[1].plot(recon["t_s"], recon["speed_mps"], label="reconstructed speed")
    axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("speed (m/s)")
    axes[1].set_title("Speed reconstruction"); axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(REPORT_DIR, "week1_feasibility_plot.png")
    plt.savefig(fig_path, dpi=130)

    print(json.dumps(result, indent=2, default=str))
    print(f"\nPlot saved -> {fig_path}")


if __name__ == "__main__":
    main()
