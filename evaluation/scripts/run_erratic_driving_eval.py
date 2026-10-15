"""
Evaluates backend/behavior/erratic_driving.py against
generate_continuous_feed.py's labeled ground truth, using the REAL
tracked trajectory (not the noise-free ground truth kinematics) --
exactly the same tracking/geometry pipeline already validated elsewhere
in this repo, applied here to a continuous feed instead of a single
triggered clip.

Metric: per-labeled-segment recall (did the detector flag *something*
overlapping each real WEAVING/HARSH_BRAKE_RELEASE segment) and
interval-level precision (of everything the detector flagged, how much
genuinely overlapped a real erratic segment) -- reported honestly, not
just "it worked."
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from run_causality_feasibility import ingest, track_vehicle
from backend.agent.loop import run_geometry_agent
from backend.behavior.erratic_driving import analyze_continuous_feed

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


GT_TO_DETECTOR_TYPE = {"WEAVING": "WEAVING", "HARSH_BRAKE_RELEASE": "HARSH_BRAKING"}


def _overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def evaluate(name="feed_1"):
    video_path = os.path.join(OUT_DIR, f"synthetic_feed_{name}.mp4")
    gt_path = os.path.join(OUT_DIR, f"synthetic_feed_{name}_ground_truth.json")
    with open(gt_path) as f:
        gt = json.load(f)

    fps, frames = ingest(video_path)
    h, w = frames[0].shape[:2]
    agent_result = run_geometry_agent(frames, w, h, 1.85, 3.0, 9.0)
    if agent_result.H is None:
        print("geometry failed -- cannot evaluate.")
        return None

    track = track_vehicle(frames, agent_result.H, fps, color="red")
    print(f"tracked {track['n_tracked']}/{len(frames)} frames "
          f"({100*track['n_tracked']/len(frames):.0f}% coverage)")

    detected = analyze_continuous_feed(track["t_s"], track["y_m"], track["speed_mps"])
    print(f"\ndetected {len(detected)} flagged interval(s):")
    for iv in detected:
        print(f"  {iv['start_s']:5.2f}s - {iv['end_s']:5.2f}s  {iv['type']:14s}  {iv['reason']}")

    erratic_gt_segments = [s for s in gt["segments"] if s["type"] != "NORMAL"]
    print(f"\nground truth has {len(erratic_gt_segments)} erratic segment(s):")
    for s in erratic_gt_segments:
        print(f"  {s['t_start_s']:5.2f}s - {s['t_end_s']:5.2f}s  {s['type']}")

    # recall: fraction of real erratic segments the detector overlapped with a matching-type flag
    hits = 0
    per_segment = []
    for s in erratic_gt_segments:
        matched = any(iv["type"] == GT_TO_DETECTOR_TYPE[s["type"]]
                      and _overlaps(iv["start_s"], iv["end_s"], s["t_start_s"], s["t_end_s"])
                      for iv in detected)
        hits += matched
        per_segment.append({"segment": s["type"], "t_start_s": s["t_start_s"], "t_end_s": s["t_end_s"], "detected": matched})
    recall = hits / len(erratic_gt_segments) if erratic_gt_segments else None

    # precision: fraction of flagged intervals that overlap a real erratic segment of the same type
    true_positives = 0
    for iv in detected:
        if any(GT_TO_DETECTOR_TYPE[s["type"]] == iv["type"]
               and _overlaps(iv["start_s"], iv["end_s"], s["t_start_s"], s["t_end_s"])
               for s in erratic_gt_segments):
            true_positives += 1
    precision = true_positives / len(detected) if detected else None

    print(f"\nrecall (real erratic segments the detector caught): {hits}/{len(erratic_gt_segments)}"
          f" = {recall:.0%}" if recall is not None else "\nrecall: n/a")
    print(f"precision (flagged intervals that were real): {true_positives}/{len(detected)}"
          f" = {precision:.0%}" if detected else "precision: n/a (nothing flagged)")

    result = {"scenario": name, "tracking_coverage": track["n_tracked"] / len(frames),
              "detected": detected, "ground_truth_erratic_segments": erratic_gt_segments,
              "per_segment_recall": per_segment, "recall": recall, "precision": precision}
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "erratic_driving_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nsaved -> {out_path}")
    return result


if __name__ == "__main__":
    evaluate("feed_1")
