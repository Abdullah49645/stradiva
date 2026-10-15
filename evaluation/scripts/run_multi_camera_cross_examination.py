"""
Multi-camera cross-examination, HARDENED version
(docs/multi_camera_architecture.md, scoped next-step #1).

Unlike the original proof of concept, this version does NOT read
`collision_frame` from ground truth to anchor synchronization. Each
camera's video is processed completely independently -- geometry,
tracking, and event detection all self-discovered from the footage
alone (backend/multiview/event_detection.py) -- and only AFTER both
cameras' independent reconstructions exist does anything compare them.
Ground truth is used only at the very end, to check the results, never
fed into the reconstruction itself.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from run_causality_feasibility import (
    ingest, track_vehicle, track_two_vehicles_bgsub, ASSUMED_LANE_HALF_WIDTH_M,
    ASSUMED_DASH_LEN_M, ASSUMED_DASH_GAP_M,
)
from backend.agent.loop import run_geometry_agent
from backend.uncertainty.propagation import propagate_speed_uncertainty
from backend.multiview.event_detection import reconstruct_with_self_discovered_truncation

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
DEGENERATE_POSITION_STD_M = 1.0


def _color_blob_pair_tracker(frames, H, fps, frame_limit):
    """Adapts the single-vehicle color_blob track_vehicle() to the
    (frames, H, fps, frame_limit) -> (track_a, track_b) shape
    reconstruct_with_self_discovered_truncation expects, matching
    track_two_vehicles_bgsub's interface so both trackers are
    interchangeable there."""
    track_a = track_vehicle(frames, H, fps, color="green", frame_limit=frame_limit)
    track_b = track_vehicle(frames, H, fps, color="red", frame_limit=frame_limit)
    return track_a, track_b


def _degenerate(track):
    x = track.get("x_m")
    return x is None or len(x) < 5 or float(np.std(x)) < DEGENERATE_POSITION_STD_M


def reconstruct_camera_independently(video_path, label):
    """Runs the FULL independent pipeline for one camera -- geometry,
    tracking, event detection -- with zero information from any other
    camera and zero ground truth. Returns a dict with everything the
    cross-examination step needs, or {"status": "..."} on failure."""
    fps, frames = ingest(video_path)
    h, w = frames[0].shape[:2]

    agent_result = run_geometry_agent(frames, w, h, ASSUMED_LANE_HALF_WIDTH_M,
                                       ASSUMED_DASH_LEN_M, ASSUMED_DASH_GAP_M)
    if agent_result.H is None:
        return {"status": "NOT_RECONSTRUCTABLE", "stop_reason": agent_result.stop_reason}
    H = agent_result.H
    if agent_result.roll_correction_deg:
        from cv.geometry.roll_correction import derotate_frame
        frames = [derotate_frame(f, agent_result.roll_correction_deg) for f in frames]

    track_a, track_b, event = reconstruct_with_self_discovered_truncation(
        frames, H, fps, track_two_vehicles_bgsub)
    used_fallback = False
    if _degenerate(track_b):
        x_b = track_b.get("x_m")
        std_display = f"{np.std(x_b):.2f}m" if x_b is not None and len(x_b) >= 2 else "insufficient data"
        print(f"  [{label}] bgsub tracker degenerate (std={std_display}) "
              f"-- falling back to color-blob tracker (see docs/multi_camera_architecture.md, "
              f"'Background-subtraction tracker failure').")
        track_a, track_b, event = reconstruct_with_self_discovered_truncation(
            frames, H, fps, _color_blob_pair_tracker)
        used_fallback = True

    if event is None:
        return {"status": "NOT_RECONSTRUCTABLE", "stop_reason": "no closest-approach event detected "
                                                                  "(vehicles never shared an overlapping track window)"}

    speed_arr = track_b.get("speed_mps", np.array([]))
    if len(speed_arr) < 6:
        return {"status": "NOT_RECONSTRUCTABLE", "stop_reason": "insufficient tracked frames for a speed estimate"}
    core = speed_arr[2:max(3, len(speed_arr) // 3)]
    speed = float(np.mean(core))

    unc = propagate_speed_uncertainty(
        agent_result.diag, {"n_tracked": track_b["n_tracked"], "n_lost": track_b["n_lost"]}, fps=fps,
        roll_correction_deg=agent_result.diag.get("roll_correction_applied_deg", 0.0))
    lo, hi = unc.interval(speed)

    return {
        "status": "RECONSTRUCTABLE", "speed_mps": speed, "ci90_lo": lo, "ci90_hi": hi,
        "event_t_local_s": event["t_local_s"], "event_gap_m": event["gap_m"],
        "used_color_blob_fallback": used_fallback, "dominant_sources": unc.dominant_sources,
        "track_b": track_b,  # raw track (t_s, x_m, ...), for coordinate reconciliation (see
                              # backend/multiview/coordinate_reconciliation.py) -- not used by
                              # the plain cross-examination flow above, only by callers that
                              # need position-level comparison, not just speed magnitude.
    }


def cross_examine_cameras(name):
    gt_path = os.path.join(OUT_DIR, f"synthetic_dualcam_{name}_ground_truth.json")
    with open(gt_path) as f:
        dual_cam_gt = json.load(f)

    report = {"scenario": name}
    results = {}
    for which in ("A", "B"):
        print(f"\n--- reconstructing camera {which} independently (no ground truth) ---")
        video_path = dual_cam_gt[f"camera_{which.lower()}"]["video"]
        r = reconstruct_camera_independently(video_path, which)
        results[which] = r
        report[f"camera_{which}"] = {k: v for k, v in r.items() if k != "track_b"}  # exclude
        # the raw numpy track from the saved JSON report -- it's only needed internally by
        # run_coordinate_reconciliation_eval.py, not useful in the frontend-facing results file
        print(f"  status={r.get('status')}", end="")
        if r.get("status") == "RECONSTRUCTABLE":
            print(f"  speed={r['speed_mps']:.2f} m/s  90% CI [{r['ci90_lo']:.2f}, {r['ci90_hi']:.2f}]"
                  f"  event_local_t={r['event_t_local_s']:.3f}s"
                  f"{'  (color-blob fallback)' if r['used_color_blob_fallback'] else ''}")
        else:
            print(f"  ({r.get('stop_reason')})")

    if any(results[w].get("status") != "RECONSTRUCTABLE" for w in ("A", "B")):
        report["sync"] = None
        report["cross_examination"] = "one or both cameras did not reconstruct -- cannot cross-examine."
        return report

    estimated_offset_b_to_a = results["A"]["event_t_local_s"] - results["B"]["event_t_local_s"]
    true_offset_b_to_a = dual_cam_gt["true_time_offset_s_B_minus_A"]
    sync_error_s = abs(estimated_offset_b_to_a - true_offset_b_to_a)
    report["sync"] = {"estimated_offset_b_to_a_s": estimated_offset_b_to_a,
                       "true_offset_b_to_a_s": true_offset_b_to_a, "sync_error_s": sync_error_s}
    print(f"\nsync (self-discovered): estimated offset (B->A) = {estimated_offset_b_to_a:+.3f}s, "
          f"true = {true_offset_b_to_a:+.3f}s, error = {sync_error_s:.3f}s")

    a, b = results["A"], results["B"]
    intervals_overlap = not (a["ci90_hi"] < b["ci90_lo"] or b["ci90_hi"] < a["ci90_lo"])
    report["cross_examination"] = {
        "camera_a_speed_mps": a["speed_mps"], "camera_b_speed_mps": b["speed_mps"],
        "intervals_overlap": intervals_overlap,
        "verdict": ("Both cameras agree: their independent 90% confidence intervals overlap."
                    if intervals_overlap else
                    "DISPUTED: the two cameras' independent 90% confidence intervals do NOT overlap."),
    }
    print(f"\ncross-examination verdict: {report['cross_examination']['verdict']}")
    return report


if __name__ == "__main__":
    report = cross_examine_cameras("collision_1")
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "multi_camera_poc_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nsaved -> {out_path}")
