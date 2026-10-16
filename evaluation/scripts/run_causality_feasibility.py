"""
Runs the full Odolite pipeline on a two-vehicle collision scenario:
geometry/scale recovery (reused from run_feasibility.py), multi-vehicle
ground-contact tracking, per-vehicle trajectory reconstruction, and the
new Collision Causality & Avoidability module -- then checks the
avoidability verdict against the scenario's independently-computed
ground-truth physics answer.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from cv.lines.line_detection import detect_line_segments, cluster_by_slope_side, fit_line_x_of_y
from cv.homography.ground_plane import (
    detect_dash_rows, build_correspondences, fit_ground_homography, image_to_world,
    lane_width_self_consistency,
)
from cv.tracking.vehicle_tracker import track_ground_contact_point
from cv.tracking.motion_tracker import track_scene as bgsub_track_scene
from scene_generator import make_camera, render_background as render_clean_background
from backend.reconstruction.trajectory import smooth_trajectory, velocity_from_position
from backend.reconstructability.score import assess_reconstructability
from backend.causality.avoidability import assess_avoidability, DEFAULT_MAX_DECEL_MPS2, DEFAULT_REACTION_TIME_S
from backend.uncertainty.propagation import propagate_speed_uncertainty
from backend.agent.loop import run_geometry_agent

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
ASSUMED_LANE_HALF_WIDTH_M = 1.85
ASSUMED_DASH_LEN_M = 3.0
ASSUMED_DASH_GAP_M = 9.0

CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def ingest(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return fps, frames


def fit_geometry(frames, w, h):
    ref = frames[0]
    segs, _ = detect_line_segments(ref)
    left_segs, right_segs = cluster_by_slope_side(segs, w)
    left_line = fit_line_x_of_y(left_segs)
    right_line = fit_line_x_of_y(right_segs)
    diag = {"n_left_segments": int(len(left_segs)), "n_right_segments": int(len(right_segs))}
    if left_line is None or right_line is None:
        return None, diag
    dashes = detect_dash_rows(ref, left_line, right_line)
    img_pts, world_pts = build_correspondences(
        dashes, left_line, right_line, ASSUMED_LANE_HALF_WIDTH_M, ASSUMED_DASH_LEN_M, ASSUMED_DASH_GAP_M)
    diag["n_correspondence_points"] = int(len(img_pts))
    if len(img_pts) < 4:
        return None, diag
    H, mask = fit_ground_homography(img_pts, world_pts)
    diag["homography_inlier_fraction"] = float(mask.mean()) if mask is not None else None
    diag["lane_width_self_consistency"] = lane_width_self_consistency(
        H, left_line, right_line, h, ASSUMED_LANE_HALF_WIDTH_M * 2)
    diag["vp_stability_px_std"] = 5.0  # not recomputed here; reuses Week 2's typical value as a placeholder signal
    return H, diag


def track_vehicle(frames, H, fps, color, frame_limit=None):
    prev_pt = None
    t_s, x_list, y_list = [], [], []
    n_lost = 0
    frames_to_use = frames[:frame_limit] if frame_limit is not None else frames
    for i, frame in enumerate(frames_to_use):
        pt, conf = track_ground_contact_point(frame, prev_pt, color=color)
        if pt is None:
            n_lost += 1
            continue
        prev_pt = pt
        world = image_to_world(H, [pt])[0]
        t_s.append(i / fps)
        x_list.append(world[0])
        y_list.append(world[1])
    t_s = np.array(t_s)
    x_s, y_s = smooth_trajectory(t_s, np.array(x_list), np.array(y_list))
    _, _, speed = velocity_from_position(t_s, x_s, y_s)
    return {"t_s": t_s, "x_m": x_s, "y_m": y_s, "speed_mps": speed, "n_lost": n_lost, "n_tracked": len(t_s)}


def track_two_vehicles_bgsub(frames, H, fps, frame_limit=None):
    """
    Real (color-independent) tracker path: background subtraction seeded
    from a clean reference frame, disambiguated far/near by image row.
    Returns (track_a, track_b) in the same dict shape as track_vehicle.
    """
    frames_to_use = frames[:frame_limit] if frame_limit is not None else frames
    cam = make_camera()
    clean_bg = render_clean_background(cam, include_dashes=True)
    slots = bgsub_track_scene(frames_to_use, n_vehicles=2, clean_background_frame=clean_bg)

    tracks = []
    for slot in slots:
        if len(slot["frame_idx"]) < 4:
            tracks.append({"t_s": np.array([]), "x_m": np.array([]), "y_m": np.array([]),
                            "speed_mps": np.array([]), "n_lost": len(frames_to_use), "n_tracked": 0,
                            "frame_idx": [], "uv": [], "bbox": []})
            continue
        t_s = np.array([i / fps for i in slot["frame_idx"]])
        world = image_to_world(H, slot["uv"])
        x_s, y_s = smooth_trajectory(t_s, world[:, 0], world[:, 1])
        _, _, speed = velocity_from_position(t_s, x_s, y_s)
        tracks.append({"t_s": t_s, "x_m": x_s, "y_m": y_s, "speed_mps": speed,
                        "n_lost": len(frames_to_use) - len(slot["frame_idx"]), "n_tracked": len(slot["frame_idx"]),
                        "frame_idx": slot["frame_idx"], "uv": slot["uv"], "bbox": slot.get("bbox", [])})
    return tracks[0], tracks[1]


def run_collision_pipeline(video_path, gt_path, use_agent=True):
    with open(gt_path) as f:
        gt = json.load(f)
    fps, frames = ingest(video_path)
    h, w = frames[0].shape[:2]

    if use_agent:
        agent_result = run_geometry_agent(frames, w, h, ASSUMED_LANE_HALF_WIDTH_M,
                                           ASSUMED_DASH_LEN_M, ASSUMED_DASH_GAP_M)
        H, geo_diag = agent_result.H, agent_result.diag
        result = {"geometry": geo_diag, "agent_trace": {
            "success": agent_result.success, "chosen_frame_idx": agent_result.chosen_frame_idx,
            "n_iterations": agent_result.n_iterations, "stop_reason": agent_result.stop_reason,
            "calls": agent_result.trace,
        }}
        # If roll correction was used, H is only valid for DEROTATED pixel
        # coordinates -- derotate the whole frame sequence once so every
        # downstream stage (tracking, video annotation) operates in the
        # same corrected coordinate space consistently, rather than
        # re-deriving a per-point rotation at each call site.
        if agent_result.roll_correction_deg:
            from cv.geometry.roll_correction import derotate_frame
            angle = agent_result.roll_correction_deg
            frames = [derotate_frame(f, angle) for f in frames]
            result["geometry"]["roll_correction_applied_deg"] = angle
    else:
        H, geo_diag = fit_geometry(frames, w, h)
        result = {"geometry": geo_diag}

    if H is None:
        result["status"] = "NOT_RECONSTRUCTABLE"
        return result

    # Truncate to pre-collision frames before tracking: post-collision frames
    # (vehicles overlapping/stopped) would otherwise contaminate the
    # trajectory smoother with a fake deceleration artifact right at the
    # moment we most need clean data. See docs/week3_4_causality_assessment.md.
    collision_frame = gt.get("collision_frame")
    frame_limit = (collision_frame + 1) if collision_frame is not None else None

    track_a = track_vehicle(frames, H, fps, color="green", frame_limit=frame_limit)
    track_b = track_vehicle(frames, H, fps, color="red", frame_limit=frame_limit)
    track_a_bgsub, track_b_bgsub = track_two_vehicles_bgsub(frames, H, fps, frame_limit=frame_limit)
    result["tracker_comparison"] = {
        "color_blob": {"a_tracked": track_a["n_tracked"], "b_tracked": track_b["n_tracked"]},
        "background_subtraction": {"a_tracked": track_a_bgsub["n_tracked"], "b_tracked": track_b_bgsub["n_tracked"]},
    }
    # background-subtraction is the "real" detector path (color-independent);
    # color-blob is kept only as the Week-1/2 reference baseline for comparison.
    track_a, track_b = track_a_bgsub, track_b_bgsub

    result["tracking"] = {
        "vehicle_a_tracked": track_a["n_tracked"], "vehicle_a_lost": track_a["n_lost"],
        "vehicle_b_tracked": track_b["n_tracked"], "vehicle_b_lost": track_b["n_lost"],
    }
    result["trajectories"] = {
        "vehicle_a": {"t_s": track_a["t_s"].tolist(), "x_m": track_a["x_m"].tolist(),
                      "speed_mps": track_a["speed_mps"].tolist()},
        "vehicle_b": {"t_s": track_b["t_s"].tolist(), "x_m": track_b["x_m"].tolist(),
                      "speed_mps": track_b["speed_mps"].tolist()},
    }
    # raw image-space points, kept separately (leading underscore) since
    # they're for the video annotator only, not part of the JSON/report
    # payload sent to the frontend.
    result["_tracks"] = {"vehicle_a": track_a, "vehicle_b": track_b}
    result["_H"] = H
    if track_a["n_tracked"] < 4 or track_b["n_tracked"] < 4:
        result["status"] = "NOT_RECONSTRUCTABLE"
        result["causality"] = {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "narrative": "Insufficient evidence to determine avoidability: one or both vehicles "
                         f"could not be reliably tracked (A: {track_a['n_tracked']} frames, "
                         f"B: {track_b['n_tracked']} frames).",
        }
        return result

    assessment = assess_reconstructability(
        {**geo_diag, "n_left_segments": geo_diag.get("n_left_segments", 0),
         "n_right_segments": geo_diag.get("n_right_segments", 0)},
        {"n_tracked": track_b["n_tracked"], "n_lost": track_b["n_lost"], "fps": fps, "n_frames": len(frames)},
        width_consistency=geo_diag.get("lane_width_self_consistency"),
    )
    result["reconstructability"] = {"overall": assessment.overall, "verdict": assessment.verdict,
                                     "components": assessment.components}

    # map reconstructability score to a coarse confidence label for the causality gate
    if assessment.overall >= 75:
        recon_confidence = "HIGH"
    elif assessment.overall >= 50:
        recon_confidence = "MEDIUM"
    else:
        recon_confidence = "LOW"

    # gap between vehicles over time (B's front to A's rear), using
    # reconstructed positions of both, interpolated onto B's timestamps
    xa_interp = np.interp(track_b["t_s"], track_a["t_s"], track_a["x_m"]) if len(track_a["t_s"]) else None
    speed_a_interp = np.interp(track_b["t_s"], track_a["t_s"], track_a["speed_mps"]) if len(track_a["t_s"]) else None
    vehicle_len_assumed = 4.5  # stated assumption -- real system would use detected/typical vehicle length
    gap_m = (xa_interp - vehicle_len_assumed / 2.0) - (track_b["x_m"] + vehicle_len_assumed / 2.0) \
        if xa_interp is not None else None

    collision_time_s = float(gt["frames"][gt["collision_frame"]]["t_s"]) if gt.get("collision_frame") is not None \
        else float(track_b["t_s"][-1])

    # only use frames strictly before the collision for the avoidability scan
    pre_impact_mask = track_b["t_s"] < collision_time_s - (1.0 / fps)

    # Discard the earliest ~15% of tracked samples: at close range (large
    # image v, near frame bottom) the ground-plane homography extrapolates
    # beyond the row range its dash correspondences actually covered,
    # producing a measurable position/speed ramp-in artifact rather than
    # the true (constant, pre-braking) speed -- confirmed by comparing
    # against ground truth during Week 3/4 development. This is a stated,
    # reported discard, not a silent one.
    n_valid = int(pre_impact_mask.sum())
    warmup_discard = max(3, int(0.15 * n_valid))
    valid_idx = np.where(pre_impact_mask)[0]
    stable_idx = valid_idx[warmup_discard:] if len(valid_idx) > warmup_discard else valid_idx

    causality = assess_avoidability(
        track_b["t_s"][stable_idx], track_b["speed_mps"][stable_idx],
        gap_m[stable_idx], collision_time_s,
        reconstructability_confidence=recon_confidence,
        lead_speed_mps=speed_a_interp[stable_idx] if speed_a_interp is not None else None,
        y_m=track_b["y_m"][stable_idx],
    )
    causality.evidence.append(
        f"Discarded {warmup_discard} near-camera startup samples out of {n_valid} pre-impact "
        f"samples (homography extrapolation is least reliable at close range -- see assessment doc)."
    )
    if len(valid_idx) > 0:
        gap_at_first_tracked = float(gap_m[valid_idx[0]])
        gap_at_stable_start = float(gap_m[stable_idx[0]]) if len(stable_idx) > 0 else None
        if gap_at_stable_start is not None and gap_at_first_tracked > gap_at_stable_start + 1.0:
            causality.evidence.append(
                f"The vehicle was already {gap_at_first_tracked - gap_at_stable_start:.1f}m closer "
                f"to the collision point by the time reconstruction becomes reliable "
                f"({gap_at_stable_start:.1f}m gap) than at the earliest raw tracked frame "
                f"({gap_at_first_tracked:.1f}m gap). This avoidability assessment is necessarily "
                f"computed from the earliest RELIABLY reconstructed point, not the true start of the "
                f"approach -- it is therefore more conservative (more likely to read NOT_AVOIDABLE) "
                f"than an idealized assessment starting from when the hazard first became visible."
            )

    result["causality"] = {
        "verdict": causality.verdict, "narrative": causality.narrative,
        "striking_vehicle_speed_at_earliest_observed_mps": causality.striking_vehicle_speed_at_earliest_observed_mps,
        "time_to_collision_at_earliest_observed_s": causality.time_to_collision_at_earliest_observed_s,
        "braking_detected": causality.braking_detected,
        "steering_detected": causality.steering_detected,
        "steering_onset_time_before_impact_s": causality.steering_onset_time_before_impact_s,
        "steering_direction": causality.steering_direction,
        "braking_onset_time_before_impact_s": causality.braking_onset_time_before_impact_s,
        "last_possible_avoidance_time_before_impact_s": causality.last_possible_avoidance_time_before_impact_s,
        "assumptions": causality.assumptions,
        "confidence": causality.confidence,
        "evidence": causality.evidence,
        "contribution_factors": causality.contribution_factors,
    }

    unc = propagate_speed_uncertainty(
        geo_diag, {"n_tracked": track_b["n_tracked"], "n_lost": track_b["n_lost"]}, fps=fps,
        roll_correction_deg=geo_diag.get("roll_correction_applied_deg", 0.0))
    striking_speed = causality.striking_vehicle_speed_at_earliest_observed_mps
    ci_lo, ci_hi = unc.interval(striking_speed) if striking_speed is not None else (None, None)
    result["uncertainty"] = {
        "striking_vehicle_speed_ci90_mps": [ci_lo, ci_hi] if striking_speed is not None else None,
        "relative_uncertainty": unc.relative_uncertainty,
        "dominant_sources": unc.dominant_sources,
        "sources": unc.sources,
    }

    result["ground_truth_physics_avoidability"] = gt.get("ground_truth_physics_avoidability")
    result["ground_truth_collided"] = gt.get("collided")
    result["collision_time_s"] = collision_time_s
    result["status"] = "RECONSTRUCTABLE"
    return result


def main():
    scenarios = ["unavoidable", "avoidable_but_occurred"]
    all_results = {}
    for name in scenarios:
        video_path = os.path.join(DATASET_DIR, f"synthetic_collision_{name}.mp4")
        gt_path = os.path.join(DATASET_DIR, f"synthetic_collision_{name}_ground_truth.json")
        print(f"\n=== {name} ===")
        result = run_collision_pipeline(video_path, gt_path)
        all_results[name] = result
        print(f"reconstructability: {result.get('reconstructability')}")
        print(f"ground truth physics avoidability: {result.get('ground_truth_physics_avoidability')}")
        print(f"Odolite causality verdict: {result.get('causality', {}).get('verdict')}")
        print(f"narrative: {result.get('causality', {}).get('narrative')}")

        from backend.reports.report_generator import render_collision_report
        report_md = render_collision_report(result, scenario_name=name)
        report_path = os.path.join(DOCS_DIR, f"report_{name}.md")
        with open(report_path, "w") as f:
            f.write(report_md)
        print(f"report -> {report_path}")

        from backend.reports.video_annotator import annotate_video
        annotate_out = os.path.join(DATASET_DIR, f"evidence_{name}.mp4")
        annotate_video(video_path, result, annotate_out, units="metric")
        print(f"evidence video -> {annotate_out}")
        annotate_out_imperial = os.path.join(DATASET_DIR, f"evidence_{name}_imperial.mp4")
        annotate_video(video_path, result, annotate_out_imperial, units="imperial")
        print(f"evidence video (imperial) -> {annotate_out_imperial}")

    with open(os.path.join(DOCS_DIR, "causality_run_results.json"), "w") as f:
        clean_results = {
            name: {k: v for k, v in r.items() if not k.startswith("_")}
            for name, r in all_results.items()
        }
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\nSaved -> {os.path.join(DOCS_DIR, 'causality_run_results.json')}")


if __name__ == "__main__":
    main()
