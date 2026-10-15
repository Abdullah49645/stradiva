"""
Multi-scenario evaluation for multi-camera cross-examination
(docs/multi_camera_architecture.md). The original evaluation ran
exactly one scenario -- honest about its numbers, but a sample size of
one is thin for a "quantitative evaluation" claim. This runs several
varied scenarios (different speeds, different sync offsets) through the
full self-discovered pipeline (geometry, tracking, event detection,
sync, coordinate reconciliation, cross-examination) and reports
aggregate statistics, not just one anecdote.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from generate_dual_camera_scene import generate_dual_camera_scene
from run_multi_camera_cross_examination import reconstruct_camera_independently
from backend.multiview.coordinate_reconciliation import reconcile

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")

SCENARIOS = [
    {"name": "sweep_v18_off0p4", "v0": 18.0, "cam_b_offset_s": 0.4},
    {"name": "sweep_v22_off0p4", "v0": 22.0, "cam_b_offset_s": 0.4},
    {"name": "sweep_v26_off0p4", "v0": 26.0, "cam_b_offset_s": 0.4},
    {"name": "sweep_v22_off0p8", "v0": 22.0, "cam_b_offset_s": 0.8},
    {"name": "sweep_v22_offm0p3", "v0": 22.0, "cam_b_offset_s": -0.3},
    {"name": "sweep_v15_off0p2", "v0": 15.0, "cam_b_offset_s": 0.2},
]


def run_one(spec):
    name = spec["name"]
    video_a, video_b, gt_path = generate_dual_camera_scene(
        name, v0=spec["v0"], cam_b_offset_s=spec["cam_b_offset_s"])
    with open(gt_path) as f:
        gt = json.load(f)

    r_a = reconstruct_camera_independently(gt["camera_a"]["video"], "A")
    r_b = reconstruct_camera_independently(gt["camera_b"]["video"], "B")
    trial = {"name": name, "v0": spec["v0"], "true_offset_s": spec["cam_b_offset_s"],
              "camera_a_status": r_a.get("status"), "camera_b_status": r_b.get("status")}

    if r_a.get("status") != "RECONSTRUCTABLE" or r_b.get("status") != "RECONSTRUCTABLE":
        trial["outcome"] = "reconstruction_failed"
        return trial

    sync_offset = r_a["event_t_local_s"] - r_b["event_t_local_s"]
    true_offset_b_to_a = gt["true_time_offset_s_B_minus_A"]
    trial["sync_error_s"] = abs(sync_offset - true_offset_b_to_a)

    a_speed, b_speed = r_a["speed_mps"], r_b["speed_mps"]
    trial["camera_a_speed_mps"] = a_speed
    trial["camera_b_speed_mps"] = b_speed
    trial["speed_agreement_pct_diff"] = 100 * abs(a_speed - b_speed) / max(a_speed, b_speed)
    intervals_overlap = not (r_a["ci90_hi"] < r_b["ci90_lo"] or r_b["ci90_hi"] < r_a["ci90_lo"])
    trial["ci_intervals_overlap"] = bool(intervals_overlap)

    track_a, track_b = r_a["track_b"], r_b["track_b"]
    recon = reconcile(track_a, track_b, r_a["event_t_local_s"], r_b["event_t_local_s"])
    if recon is None:
        trial["outcome"] = "reconciliation_failed"
        return trial

    t_b_on_a_clock = np.asarray(track_b["t_s"]) + sync_offset
    t_lo = max(track_a["t_s"][0], t_b_on_a_clock.min())
    t_hi = min(track_a["t_s"][-1], t_b_on_a_clock.max())
    if t_hi <= t_lo:
        trial["outcome"] = "no_overlap"
        return trial

    t_common = np.linspace(t_lo, t_hi, max(10, int((t_hi - t_lo) * 20)))
    x_a_common = np.interp(t_common, track_a["t_s"], track_a["x_m"])
    x_b_common = np.interp(t_common, t_b_on_a_clock, recon["track_b_reconciled_x_m"])
    trial["mean_position_error_m"] = float(np.abs(x_a_common - x_b_common).mean())
    trial["direction_sign"] = recon["direction_sign"]
    trial["outcome"] = "success"
    return trial


def main():
    trials = [run_one(s) for s in SCENARIOS]
    successes = [t for t in trials if t["outcome"] == "success"]

    summary = {
        "n_scenarios": len(trials),
        "n_success": len(successes),
        "mean_sync_error_s": float(np.mean([t["sync_error_s"] for t in successes])) if successes else None,
        "mean_speed_agreement_pct_diff": float(np.mean([t["speed_agreement_pct_diff"] for t in successes])) if successes else None,
        "ci_overlap_rate": float(np.mean([t["ci_intervals_overlap"] for t in successes])) if successes else None,
        "mean_position_error_m": float(np.mean([t["mean_position_error_m"] for t in successes])) if successes else None,
        "direction_sign_correct_rate": float(np.mean([t["direction_sign"] == -1 for t in successes])) if successes else None,
        "trials": trials,
    }

    print(f"\n{len(successes)}/{len(trials)} scenarios succeeded end-to-end")
    for t in trials:
        if t["outcome"] == "success":
            print(f"  {t['name']}: sync_err={t['sync_error_s']:.3f}s  "
                  f"speed_agree={t['speed_agreement_pct_diff']:.1f}%  "
                  f"pos_err={t['mean_position_error_m']:.2f}m  "
                  f"CI_overlap={t['ci_intervals_overlap']}")
        else:
            print(f"  {t['name']}: {t['outcome']}")

    if successes:
        print(f"\nmean sync error: {summary['mean_sync_error_s']:.3f}s")
        print(f"mean speed agreement (pct diff between cameras): {summary['mean_speed_agreement_pct_diff']:.1f}%")
        print(f"CI overlap rate (both cameras statistically agree): {summary['ci_overlap_rate']:.0%}")
        print(f"mean position error after reconciliation: {summary['mean_position_error_m']:.2f}m")
        print(f"direction sign correctly detected: {summary['direction_sign_correct_rate']:.0%}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "multi_camera_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
