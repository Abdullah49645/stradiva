"""
Runs the full pipeline (run_feasibility.run_pipeline) against every
generated scenario and prints a comparison table -- this is the check
that the reconstructability gate behaves correctly across more than one
hand-tuned case (brief sections 24/26).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from run_feasibility import run_pipeline, DATASET_DIR

SCENARIOS = {
    "A (baseline)": "A",
    "B (no scale reference)": "B_no_scale_reference",
    "C (8deg camera roll)": "C_camera_roll",
    "D (heavy fog)": "D_heavy_fog",
    "E (faster vehicle, 32 m/s)": "E_faster_vehicle",
}

EXPECTATION = {
    "A (baseline)": "RECONSTRUCTABLE",
    "B (no scale reference)": "NOT_RECONSTRUCTABLE",
    "C (8deg camera roll)": "RECONSTRUCTABLE_LOW_CONFIDENCE or worse (assumption violated)",
    "D (heavy fog)": "NOT_RECONSTRUCTABLE",
    "E (faster vehicle, 32 m/s)": "RECONSTRUCTABLE",
}


def main():
    rows = []
    for label, key in SCENARIOS.items():
        video_path = os.path.join(DATASET_DIR, f"synthetic_scene_{key}.mp4")
        gt_path = os.path.join(DATASET_DIR, f"synthetic_scene_{key}_ground_truth.json")
        if not os.path.exists(video_path):
            print(f"SKIP {label}: {video_path} not found (run generate_stress_scenes.py first)")
            continue
        result = run_pipeline(video_path, gt_path)
        verdict = result["status"]
        overall = result.get("reconstructability", {}).get("overall")
        speed_err = result.get("scores", {}).get("speed_error_pct")
        disp_mae = result.get("scores", {}).get("displacement_mae_m")
        reasons = result.get("reconstructability", {}).get("reasons", [])
        rows.append({
            "scenario": label, "expected": EXPECTATION[label], "verdict": verdict,
            "score": overall, "speed_error_pct": speed_err, "displacement_mae_m": disp_mae,
            "reasons": reasons,
        })

    print(f"\n{'scenario':32s} {'expected':45s} {'verdict':30s} {'score':>6s} {'spd_err%':>9s} {'disp_mae':>9s}")
    for r in rows:
        print(f"{r['scenario']:32s} {r['expected']:45s} {r['verdict']:30s} "
              f"{str(r['score']):>6s} {str(r['speed_error_pct']):>9s} {str(r['displacement_mae_m']):>9s}")
        for reason in r["reasons"]:
            print(f"    - {reason}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "week2_stress_test_results.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
