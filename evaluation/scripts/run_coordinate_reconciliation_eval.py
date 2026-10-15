"""
Coordinate reconciliation evaluation (docs/multi_camera_architecture.md,
scoped next-step #2).

Builds on run_multi_camera_cross_examination.py's already-validated
pieces (independent geometry/tracking per camera, self-discovered event
detection, self-discovered time sync) and adds the piece that was
explicitly flagged as missing: comparing POSITION, not just speed
magnitude, between two independently-calibrated cameras.

Method:
  1. Reconstruct each camera fully independently (reused, not
     reimplemented).
  2. backend/multiview/coordinate_reconciliation.py determines the
     direction sign + offset needed to express camera B's positions in
     camera A's coordinate frame, anchored at each camera's own
     self-detected event time (no ground truth).
  3. Map camera B's local timeline onto camera A's using the
     already-validated sync offset.
  4. At overlapping synced times, compare camera A's own reconstructed
     position against camera B's reconciled position for the SAME
     physical vehicle -- if reconciliation is doing its job, these
     should now agree; if it isn't, this is exactly where that would
     show up.

Reports the honest position error, not just whether reconciliation ran
without crashing.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from run_multi_camera_cross_examination import reconstruct_camera_independently
from backend.multiview.coordinate_reconciliation import reconcile

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


def evaluate(name="collision_1"):
    gt_path = os.path.join(OUT_DIR, f"synthetic_dualcam_{name}_ground_truth.json")
    with open(gt_path) as f:
        gt = json.load(f)

    print("--- reconstructing camera A independently ---")
    r_a = reconstruct_camera_independently(gt["camera_a"]["video"], "A")
    print("--- reconstructing camera B independently ---")
    r_b = reconstruct_camera_independently(gt["camera_b"]["video"], "B")

    if r_a.get("status") != "RECONSTRUCTABLE" or r_b.get("status") != "RECONSTRUCTABLE":
        print("one or both cameras did not reconstruct -- cannot evaluate reconciliation.")
        return None

    track_a, track_b = r_a["track_b"], r_b["track_b"]  # "track_b" key = the striking vehicle's
                                                        # track within EACH camera's own result
    recon = reconcile(track_a, track_b, r_a["event_t_local_s"], r_b["event_t_local_s"])
    if recon is None:
        print("reconciliation failed -- event time outside tracked window for one camera.")
        return None

    print(f"\ndirection_sign applied to camera B: {recon['direction_sign']:+d}")
    print(f"offset applied to camera B: {recon['offset_m']:.2f}m")
    print(f"anchor check -- camera A position at shared event: {recon['x_a_at_event_m']:.2f}m, "
          f"camera B reconciled position at its own shared event: {recon['x_b_at_event_reconciled_m']:.2f}m "
          f"(should match by construction, confirms the arithmetic, not yet the real test)")

    # map camera B's local time onto camera A's timeline using the same
    # sync offset validated in run_multi_camera_cross_examination.py
    sync_offset_b_to_a = r_a["event_t_local_s"] - r_b["event_t_local_s"]
    t_b_on_a_clock = np.asarray(track_b["t_s"]) + sync_offset_b_to_a

    t_lo = max(track_a["t_s"][0], t_b_on_a_clock.min())
    t_hi = min(track_a["t_s"][-1], t_b_on_a_clock.max())
    if t_hi <= t_lo:
        print("no overlapping synced time window between the two cameras -- cannot compare positions.")
        return None

    t_common = np.linspace(t_lo, t_hi, max(20, int((t_hi - t_lo) * 20)))
    x_a_common = np.interp(t_common, track_a["t_s"], track_a["x_m"])
    x_b_common = np.interp(t_common, t_b_on_a_clock, recon["track_b_reconciled_x_m"])
    position_error_m = np.abs(x_a_common - x_b_common)

    # also compare against the UNRECONCILED camera B positions, to show
    # reconciliation is actually doing something, not a no-op
    x_b_common_unreconciled = np.interp(t_common, t_b_on_a_clock, track_b["x_m"])
    position_error_unreconciled_m = np.abs(x_a_common - x_b_common_unreconciled)

    result = {
        "scenario": name,
        "direction_sign": recon["direction_sign"],
        "offset_m": recon["offset_m"],
        "overlapping_synced_window_s": [float(t_lo), float(t_hi)],
        "mean_position_error_m_after_reconciliation": float(position_error_m.mean()),
        "max_position_error_m_after_reconciliation": float(position_error_m.max()),
        "mean_position_error_m_WITHOUT_reconciliation": float(position_error_unreconciled_m.mean()),
    }
    print(f"\noverlapping synced window: {t_lo:.2f}s - {t_hi:.2f}s ({len(t_common)} comparison points)")
    print(f"mean position error AFTER reconciliation:  {result['mean_position_error_m_after_reconciliation']:.2f}m")
    print(f"max position error AFTER reconciliation:   {result['max_position_error_m_after_reconciliation']:.2f}m")
    print(f"mean position error WITHOUT reconciliation: {result['mean_position_error_m_WITHOUT_reconciliation']:.2f}m "
          f"(for comparison -- shows reconciliation is doing real work, not a no-op)")

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "coordinate_reconciliation_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nsaved -> {out_path}")
    return result


if __name__ == "__main__":
    evaluate("collision_1")
