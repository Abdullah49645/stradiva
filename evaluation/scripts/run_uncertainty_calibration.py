"""
Uncertainty calibration sweep (brief section 25's "confidence interval
coverage" metric).

Generates a grid of synthetic single-vehicle scenes with varied speed
and camera roll (the two parameters scene_generator.py exposes), runs
each through the real pipeline (agentic geometry search + roll
correction + tracking + speed reconstruction), computes the stated 90%
CI from backend/uncertainty/propagation.py, and checks whether the
scene's true (ground-truth) speed actually falls inside that interval.

Reports empirical coverage: for a correctly-calibrated 90% interval,
this should land near 90% as the trial count grows. Also does a small
1D search over CI90_SCALE to find the value that gets closest to 90%
coverage on this trial set, and writes that as the recommended constant
-- propagation.py's CI90_SCALE should be updated to match by hand after
reviewing the result (kept as a manual step so a bad automated fit can't
silently degrade the stated intervals).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scene_generator import generate_scene
from run_feasibility import run_pipeline
from backend.uncertainty.propagation import propagate_speed_uncertainty

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
TMP_DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "_calibration_tmp")
os.makedirs(TMP_DATASET_DIR, exist_ok=True)

SPEEDS_MPS = [12.0, 18.0, 24.0, 30.0]
ROLL_DEGS = [0.0, -3.0, 3.0, -6.0, 6.0, -9.0, 9.0]


def run_trial(name, speed_mps, roll_deg):
    video_path, gt_path = generate_scene(
        f"calib_{name}", roll_deg=roll_deg, speed_mps=speed_mps)
    result = run_pipeline(video_path=video_path, gt_path=gt_path, save_plot=False)
    with open(gt_path) as f:
        gt = json.load(f)
    true_speed = gt["vehicle"]["speed_true_mps"]

    trial = {"name": name, "speed_true_mps": true_speed, "roll_deg": roll_deg,
              "status": result.get("status")}
    if result.get("status") != "RECONSTRUCTABLE" or "scores" not in result:
        trial["reconstructable"] = False
        return trial

    trial["reconstructable"] = True
    est_speed = result["scores"]["speed_estimated_mean_mps"]
    trial["speed_estimated_mps"] = est_speed

    unc = propagate_speed_uncertainty(
        result["geometry"], result["tracking"],
        fps=30.0, roll_correction_deg=result["geometry"].get("roll_correction_applied_deg", 0.0))
    lo, hi = unc.interval(est_speed)
    trial["ci90_lo_mps"] = lo
    trial["ci90_hi_mps"] = hi
    trial["ci90_half_width_frac"] = unc.ci90_half_width_frac
    trial["dominant_sources"] = unc.dominant_sources
    trial["covered"] = bool(lo <= true_speed <= hi)
    return trial


def coverage_at_scale(trials, scale):
    """Re-derive covered/not-covered at a hypothetical CI90_SCALE without
    re-running the whole pipeline. Trials were run with CI90_SCALE=1.0,
    so each trial's stored ci90_half_width_frac already equals the raw
    (pre-scale) combined relative uncertainty; the interval scales
    linearly in that value, so we just reapply the candidate scale."""
    n, covered = 0, 0
    for t in trials:
        if not t.get("reconstructable"):
            continue
        raw_relative_uncertainty = t["ci90_half_width_frac"]
        half_width = t["speed_estimated_mps"] * raw_relative_uncertainty * scale
        lo = max(0.0, t["speed_estimated_mps"] - half_width)
        hi = t["speed_estimated_mps"] + half_width
        n += 1
        covered += int(lo <= t["speed_true_mps"] <= hi)
    return covered / n if n else None


def main():
    trials = []
    for speed in SPEEDS_MPS:
        for roll in ROLL_DEGS:
            name = f"s{int(speed)}_r{roll:+.0f}".replace("+", "p").replace("-", "m")
            print(f"running trial {name} ...")
            trials.append(run_trial(name, speed, roll))

    reconstructable = [t for t in trials if t.get("reconstructable")]
    covered = [t for t in reconstructable if t.get("covered")]
    coverage = len(covered) / len(reconstructable) if reconstructable else None

    scale_sweep = {}
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        scale_sweep[scale] = coverage_at_scale(reconstructable, scale)
    best_scale = min(scale_sweep, key=lambda s: abs((scale_sweep[s] or 0) - 0.90))

    summary = {
        "n_trials": len(trials),
        "n_reconstructable": len(reconstructable),
        "n_not_reconstructable": len(trials) - len(reconstructable),
        "coverage_at_scale_1.0": coverage,
        "target_coverage": 0.90,
        "scale_sweep_coverage": scale_sweep,
        "recommended_ci90_scale": best_scale,
        "trials": trials,
    }
    out_path = os.path.join(DOCS_DIR, "uncertainty_calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nn_reconstructable={len(reconstructable)}/{len(trials)}")
    print(f"coverage @ CI90_SCALE=1.0: {coverage}")
    print("coverage across candidate scales:")
    for s, c in scale_sweep.items():
        print(f"  scale={s}: coverage={c}")
    print(f"recommended CI90_SCALE: {best_scale}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
