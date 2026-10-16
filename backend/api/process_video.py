"""
Production entry point: process_video(path) runs the full Odolite pipeline
on an arbitrary video file, with NO dependency on synthetic-scene
internals (unlike evaluation/scripts/run_causality_feasibility.py, which
uses the known synthetic camera model to render a perfectly clean
background reference -- not available for real footage).

This is what infrastructure/aws/lambda_handler.py calls.
"""
import json
import os

import cv2
import numpy as np

from cv.lines.line_detection import detect_line_segments, cluster_by_slope_side, fit_line_x_of_y
from cv.homography.ground_plane import (
    detect_dash_rows, build_correspondences, fit_ground_homography, image_to_world,
    lane_width_self_consistency,
)
from cv.tracking.motion_tracker import BackgroundSubtractionTracker
from backend.reconstruction.trajectory import smooth_trajectory, velocity_from_position
from backend.reconstructability.score import assess_reconstructability
from backend.causality.avoidability import assess_avoidability
from backend.agent.loop import run_geometry_agent
from backend.reports.report_generator import render_collision_report
from backend.uncertainty.propagation import propagate_speed_uncertainty

ASSUMED_LANE_HALF_WIDTH_M = 1.85
ASSUMED_DASH_LEN_M = 3.0
ASSUMED_DASH_GAP_M = 9.0
COLLISION_GAP_THRESHOLD_M = 1.5  # gap below this -> treat as contact/collision frame


def ingest_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return fps, frames


def estimate_background(frames, n_samples=20):
    """
    Temporal-median background estimate: works for real footage without
    any known camera model, PROVIDED no object sits in the same pixels
    for more than ~half the sampled frames. A vehicle that is stationary
    for nearly the entire clip (e.g. already stopped when the clip
    starts, as in our 'unavoidable'/'avoidable_but_occurred' synthetic
    stress scenarios) will bleed into this estimate -- a genuine,
    currently-unsolved limitation of passive background estimation, not
    papered over here. See docs/aws_and_production_notes.md.
    """
    n = len(frames)
    idxs = np.linspace(0, n - 1, min(n_samples, n)).astype(int)
    stack = np.stack([frames[i] for i in idxs], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def track_two_vehicles(frames, H, fps, background_frame, n_vehicles=2, warmup_frames=10):
    tracker = BackgroundSubtractionTracker(min_area=25)
    tracker.seed_background(background_frame)
    slots = [{"frame_idx": [], "uv": []} for _ in range(n_vehicles)]
    for i, frame in enumerate(frames):
        blobs = tracker.detect_blobs(frame)
        if len(blobs) < n_vehicles:
            continue
        top = sorted(blobs, key=lambda b: -b["area"])[:n_vehicles]
        top.sort(key=lambda b: b["bottom_center"][1])
        for slot_idx, blob in enumerate(top):
            slots[slot_idx]["frame_idx"].append(i)
            slots[slot_idx]["uv"].append(blob["bottom_center"])

    tracks = []
    for slot in slots:
        if len(slot["frame_idx"]) < 4:
            tracks.append({"t_s": np.array([]), "x_m": np.array([]), "y_m": np.array([]),
                            "speed_mps": np.array([]), "n_tracked": 0,
                            "n_lost": len(frames) - len(slot["frame_idx"])})
            continue
        t_s = np.array([i / fps for i in slot["frame_idx"]])
        world = image_to_world(H, slot["uv"])
        x_s, y_s = smooth_trajectory(t_s, world[:, 0], world[:, 1])
        _, _, speed = velocity_from_position(t_s, x_s, y_s)
        tracks.append({"t_s": t_s, "x_m": x_s, "y_m": y_s, "speed_mps": speed,
                        "n_tracked": len(slot["frame_idx"]),
                        "n_lost": len(frames) - len(slot["frame_idx"])})
    return tracks


def detect_collision(track_a, track_b, vehicle_len_assumed=4.5, gap_threshold_m=COLLISION_GAP_THRESHOLD_M):
    """Heuristic collision-frame detector: first time the reconstructed gap
    between vehicles drops below threshold. Returns (collision_time_s, gap_array, t_common) or (None, None, None)."""
    if len(track_a["t_s"]) < 2 or len(track_b["t_s"]) < 2:
        return None, None, None
    t_common = track_b["t_s"]
    xa_interp = np.interp(t_common, track_a["t_s"], track_a["x_m"])
    gap = (xa_interp - vehicle_len_assumed / 2.0) - (track_b["x_m"] + vehicle_len_assumed / 2.0)
    below = np.where(gap <= gap_threshold_m)[0]
    if len(below) == 0:
        return None, gap, t_common
    return float(t_common[below[0]]), gap, t_common


def process_video(video_path, n_vehicles_hint=2, scenario_name=None):
    """Main production entry point. Returns (result_dict, report_markdown)."""
    scenario_name = scenario_name or os.path.basename(video_path)
    fps, frames = ingest_video(video_path)
    if len(frames) < 5:
        result = {"status": "NOT_RECONSTRUCTABLE",
                  "reconstructability": {"reasons": ["fewer than 5 frames decoded from input"]}}
        return result, render_collision_report(result, scenario_name)

    h, w = frames[0].shape[:2]

    agent_result = run_geometry_agent(frames, w, h, ASSUMED_LANE_HALF_WIDTH_M,
                                       ASSUMED_DASH_LEN_M, ASSUMED_DASH_GAP_M)
    result = {"geometry": agent_result.diag, "agent_trace": {
        "success": agent_result.success, "chosen_frame_idx": agent_result.chosen_frame_idx,
        "n_iterations": agent_result.n_iterations, "stop_reason": agent_result.stop_reason,
        "calls": agent_result.trace,
    }}
    if agent_result.H is None:
        result["status"] = "NOT_RECONSTRUCTABLE"
        return result, render_collision_report(result, scenario_name)

    # If the agent found roll correction necessary, H was fit against a
    # DEROTATED frame (see cv/geometry/roll_correction.py) -- every frame
    # used downstream (background estimation, tracking) must be derotated
    # by the same angle or the recovered homography no longer matches the
    # pixel coordinates being fed into it. This mirrors what
    # evaluation/scripts/run_causality_feasibility.py already does; it was
    # missing here, so real (non-synthetic) roll-corrected footage would
    # have silently produced a wrong reconstruction despite the agent
    # reporting success.
    if agent_result.roll_correction_deg:
        from cv.geometry.roll_correction import derotate_frame
        angle = agent_result.roll_correction_deg
        frames = [derotate_frame(f, angle) for f in frames]
        result["geometry"]["roll_correction_applied_deg"] = angle

    H = agent_result.H
    background = estimate_background(frames)
    tracks = track_two_vehicles(frames, H, fps, background, n_vehicles=n_vehicles_hint)
    track_a, track_b = tracks[0], tracks[1] if len(tracks) > 1 else {"n_tracked": 0, "n_lost": len(frames)}

    result["tracking"] = {
        "vehicle_a_tracked": track_a["n_tracked"], "vehicle_a_lost": track_a["n_lost"],
        "vehicle_b_tracked": track_b["n_tracked"], "vehicle_b_lost": track_b["n_lost"],
    }
    if track_a["n_tracked"] < 4 or track_b["n_tracked"] < 4:
        result["status"] = "NOT_RECONSTRUCTABLE"
        result["causality"] = {"verdict": "INSUFFICIENT_EVIDENCE",
                                "narrative": "One or both vehicles could not be reliably tracked."}
        return result, render_collision_report(result, scenario_name)

    assessment = assess_reconstructability(
        agent_result.diag,
        {"n_tracked": track_b["n_tracked"], "n_lost": track_b["n_lost"], "fps": fps, "n_frames": len(frames)},
        width_consistency=agent_result.diag.get("lane_width_self_consistency"),
    )
    result["reconstructability"] = {"overall": assessment.overall, "verdict": assessment.verdict,
                                     "components": assessment.components}
    recon_confidence = "HIGH" if assessment.overall >= 75 else ("MEDIUM" if assessment.overall >= 50 else "LOW")

    collision_time_s, gap, t_common = detect_collision(track_a, track_b)
    if collision_time_s is None:
        result["status"] = assessment.verdict
        result["causality"] = {"verdict": "INSUFFICIENT_EVIDENCE",
                                "narrative": "No collision point detected between the two tracked vehicles "
                                             "in the observed window; avoidability analysis requires an "
                                             "identifiable collision point."}
        return result, render_collision_report(result, scenario_name)

    pre_impact_mask = t_common < collision_time_s - (1.0 / fps)
    n_valid = int(pre_impact_mask.sum())
    warmup_discard = max(3, int(0.15 * n_valid)) if n_valid > 0 else 0
    valid_idx = np.where(pre_impact_mask)[0]
    stable_idx = valid_idx[warmup_discard:] if len(valid_idx) > warmup_discard else valid_idx

    causality = assess_avoidability(
        t_common[stable_idx], track_b["speed_mps"][stable_idx], gap[stable_idx],
        collision_time_s, reconstructability_confidence=recon_confidence,
    )

    # Uncertainty (brief section 15): turn the diagnostics already
    # computed above into a stated 90% CI on the striking vehicle's
    # earliest-observed speed, rather than reporting it as a bare point
    # estimate. See backend/uncertainty/propagation.py for the method
    # and its calibration.
    unc = propagate_speed_uncertainty(
        agent_result.diag, {"n_tracked": track_b["n_tracked"], "n_lost": track_b["n_lost"]}, fps=fps,
        roll_correction_deg=agent_result.diag.get("roll_correction_applied_deg", 0.0))
    striking_speed = causality.striking_vehicle_speed_at_earliest_observed_mps
    ci_lo, ci_hi = unc.interval(striking_speed) if striking_speed is not None else (None, None)
    result["uncertainty"] = {
        "striking_vehicle_speed_ci90_mps": [ci_lo, ci_hi] if striking_speed is not None else None,
        "relative_uncertainty": unc.relative_uncertainty,
        "dominant_sources": unc.dominant_sources,
        "sources": unc.sources,
    }

    result["causality"] = {
        "verdict": causality.verdict, "narrative": causality.narrative,
        "striking_vehicle_speed_at_earliest_observed_mps": causality.striking_vehicle_speed_at_earliest_observed_mps,
        "time_to_collision_at_earliest_observed_s": causality.time_to_collision_at_earliest_observed_s,
        "braking_detected": causality.braking_detected,
        "braking_onset_time_before_impact_s": causality.braking_onset_time_before_impact_s,
        "last_possible_avoidance_time_before_impact_s": causality.last_possible_avoidance_time_before_impact_s,
        "assumptions": causality.assumptions, "confidence": causality.confidence,
        "evidence": causality.evidence,
    }
    result["status"] = assessment.verdict
    return result, render_collision_report(result, scenario_name)
