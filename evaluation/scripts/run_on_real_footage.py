"""
Real-footage harness (brief sections 24-27: evaluation must include real
footage, not just synthetic scenes -- and correctly rejecting genuinely
unmeasurable video is as important a result as a correct reconstruction).

Every scenario tested so far (run_stress_tests.py, run_causality_feasibility.py,
run_uncertainty_calibration.py) is synthetic. This script is the other half:
point it at ANY real video (dashcam, CCTV, phone footage) and it runs the
exact same production pipeline -- agentic geometry search, roll correction,
tracking, reconstructability gate, speed + uncertainty -- with no ground
truth required. If you happen to know the true speed independently (a
GPS-logged test drive, a radar-gun clip, etc.) pass --true-speed-mps and
it'll report error against that too; otherwise it just reports what the
pipeline concludes, honestly, including a straight rejection if the
footage can't support a quantitative estimate.

This is deliberately a THIN wrapper around run_feasibility.py's stage
functions -- it does not reimplement any CV logic, just removes the
synthetic-ground-truth assumption baked into run_pipeline().

Usage:
    python run_on_real_footage.py path/to/clip.mp4
    python run_on_real_footage.py path/to/clip.mp4 --true-speed-mps 15.6
    python run_on_real_footage.py               # runs every video already
                                                  # dropped into
                                                  # evaluation/dataset/real/
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from run_feasibility import (
    stage1_ingest, stage3_track_and_reconstruct,
    ASSUMED_LANE_HALF_WIDTH_M, ASSUMED_DASH_LEN_M, ASSUMED_DASH_GAP_M,
)
from backend.agent.loop import run_geometry_agent
from backend.reconstructability.score import assess_reconstructability
from backend.uncertainty.propagation import propagate_speed_uncertainty
from cv.geometry.roll_correction import derotate_frame

REAL_DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "real")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs")


def run_on_video(video_path, true_speed_mps=None):
    print(f"\n=== {os.path.basename(video_path)} ===")
    ingest_info, frames = stage1_ingest(video_path)
    if not frames:
        print("  Could not read any frames -- unsupported codec/container or corrupt file.")
        return {"video": video_path, "status": "INGEST_FAILED"}

    agent_result = run_geometry_agent(frames, ingest_info["w"], ingest_info["h"],
                                       ASSUMED_LANE_HALF_WIDTH_M, ASSUMED_DASH_LEN_M, ASSUMED_DASH_GAP_M)
    geo_diag = agent_result.diag
    result = {
        "video": video_path, "ingest": ingest_info, "geometry": geo_diag,
        "agent_trace": {"success": agent_result.success, "chosen_frame_idx": agent_result.chosen_frame_idx,
                         "n_iterations": agent_result.n_iterations, "stop_reason": agent_result.stop_reason,
                         "roll_correction_deg": agent_result.roll_correction_deg},
    }
    if agent_result.roll_correction_deg:
        frames = [derotate_frame(f, agent_result.roll_correction_deg) for f in frames]
        geo_diag["roll_correction_applied_deg"] = agent_result.roll_correction_deg

    if agent_result.H is None:
        assessment = assess_reconstructability(
            geo_diag, {"n_tracked": 0, "n_lost": 0, "fps": ingest_info["fps"], "n_frames": ingest_info["n_frames"]})
        result["reconstructability"] = {"overall": assessment.overall, "verdict": assessment.verdict,
                                         "reasons": assessment.reasons}
        result["status"] = "NOT_RECONSTRUCTABLE"
        print(f"  REJECTED: {assessment.verdict} (score {assessment.overall})")
        for r in assessment.reasons:
            print(f"    - {r}")
        print(f"  stop_reason: {agent_result.stop_reason}")
        return result

    recon = stage3_track_and_reconstruct(frames, agent_result.H, ingest_info["fps"])
    result["tracking"] = {"n_tracked": recon["n_tracked"], "n_lost": recon["n_lost"]}

    assessment = assess_reconstructability(
        geo_diag, {"n_tracked": recon["n_tracked"], "n_lost": recon["n_lost"],
                   "fps": ingest_info["fps"], "n_frames": ingest_info["n_frames"]},
        width_consistency=geo_diag.get("lane_width_self_consistency"))
    result["reconstructability"] = {"overall": assessment.overall, "verdict": assessment.verdict,
                                     "reasons": assessment.reasons}
    result["status"] = assessment.verdict

    if len(recon["speed_mps"]) == 0 or assessment.verdict == "NOT_RECONSTRUCTABLE":
        print(f"  REJECTED: {assessment.verdict} (score {assessment.overall}) despite a fitted homography "
              f"-- tracking/consistency checks failed.")
        for r in assessment.reasons:
            print(f"    - {r}")
        return result

    core = slice(len(recon["speed_mps"]) // 6, -max(1, len(recon["speed_mps"]) // 6))
    speed_core = recon["speed_mps"][core]
    est_speed = float(speed_core.mean()) if len(speed_core) else None
    result["speed_estimated_mean_mps"] = est_speed

    if est_speed is not None:
        unc = propagate_speed_uncertainty(
            geo_diag, result["tracking"], fps=ingest_info["fps"],
            roll_correction_deg=geo_diag.get("roll_correction_applied_deg", 0.0))
        lo, hi = unc.interval(est_speed)
        result["uncertainty"] = {
            "ci90_mps": [lo, hi], "relative_uncertainty": unc.relative_uncertainty,
            "dominant_sources": unc.dominant_sources,
        }
        print(f"  {assessment.verdict} (score {assessment.overall})")
        print(f"  estimated speed: {est_speed:.1f} m/s  (90% CI {lo:.1f}-{hi:.1f} m/s)")
        print(f"  {est_speed*3.6:.0f} km/h  (90% CI {lo*3.6:.0f}-{hi*3.6:.0f} km/h)")
        print(f"  dominant uncertainty: {', '.join(unc.dominant_sources) or 'n/a'}")
        if true_speed_mps is not None:
            err_pct = 100 * abs(est_speed - true_speed_mps) / true_speed_mps
            covered = lo <= true_speed_mps <= hi
            result["true_speed_mps"] = true_speed_mps
            result["speed_error_pct"] = err_pct
            result["ci_covered_true_speed"] = covered
            print(f"  true speed: {true_speed_mps:.1f} m/s  |  error: {err_pct:.1f}%  |  "
                  f"CI covers true value: {covered}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default=None,
                         help=f"Path to a real video. If omitted, runs every video in {REAL_DATASET_DIR}/")
    parser.add_argument("--true-speed-mps", type=float, default=None,
                         help="Independently-known true speed, if you have one (e.g. GPS-logged test drive).")
    args = parser.parse_args()

    if args.video:
        video_paths = [args.video]
    else:
        video_paths = sorted(
            glob.glob(os.path.join(REAL_DATASET_DIR, "*.mp4")) +
            glob.glob(os.path.join(REAL_DATASET_DIR, "*.mov")) +
            glob.glob(os.path.join(REAL_DATASET_DIR, "*.avi")))
        if not video_paths:
            print(f"No videos found in {REAL_DATASET_DIR}/. Drop a real clip there "
                  f"(see {REAL_DATASET_DIR}/README.md) or pass a path directly.")
            return

    results = [run_on_video(vp, true_speed_mps=args.true_speed_mps) for vp in video_paths]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "real_footage_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
