"""
Multi-scenario evaluation for erratic-driving detection
(docs/erratic_driving_detection.md). The original evaluation ran one
scenario. This runs several -- varied weave amplitude, varied brake
severity, and critically a PURE-NORMAL scenario with zero erratic
segments, which the original evaluation never tested: does the detector
false-alarm on completely ordinary driving? A system that cries wolf on
benign footage is not usable regardless of its recall/precision on
erratic segments, so this is not optional.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from generate_continuous_feed import generate_continuous_feed
from run_erratic_driving_eval import evaluate, GT_TO_DETECTOR_TYPE

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")

SCENARIOS = {
    "sweep_baseline": None,  # use generate_continuous_feed's own default segments
    "sweep_strong_signal": [
        {"type": "NORMAL", "duration_s": 1.0, "v0": 7.0},
        {"type": "WEAVING", "duration_s": 2.0, "amplitude_m": 1.1, "period_s": 1.6},
        {"type": "NORMAL", "duration_s": 1.0},
        {"type": "HARSH_BRAKE_RELEASE", "duration_s": 1.8, "decel_mps2": 6.5, "brake_duration_s": 0.8},
        {"type": "NORMAL", "duration_s": 1.2},
    ],
    "sweep_weak_signal": [
        {"type": "NORMAL", "duration_s": 1.0, "v0": 7.0},
        {"type": "WEAVING", "duration_s": 2.0, "amplitude_m": 0.35, "period_s": 1.6},
        {"type": "NORMAL", "duration_s": 1.0},
        {"type": "HARSH_BRAKE_RELEASE", "duration_s": 1.8, "decel_mps2": 3.2, "brake_duration_s": 0.7},
        {"type": "NORMAL", "duration_s": 1.2},
    ],
    "sweep_pure_normal": [
        {"type": "NORMAL", "duration_s": 2.0, "v0": 7.0},
        {"type": "NORMAL", "duration_s": 2.0},
        {"type": "NORMAL", "duration_s": 2.0},
        {"type": "NORMAL", "duration_s": 2.0},
    ],
}


def main():
    results = {}
    for name, segments in SCENARIOS.items():
        print(f"\n=== {name} ===")
        generate_continuous_feed(name, segments=segments)
        r = evaluate(name)
        results[name] = r

    print("\n\n=== SWEEP SUMMARY ===")
    for name, r in results.items():
        if r is None:
            print(f"{name}: evaluation failed (geometry/tracking)")
            continue
        n_detected = len(r["detected"])
        n_gt = len(r["ground_truth_erratic_segments"])
        if n_gt == 0:
            print(f"{name} (pure normal, 0 real erratic segments): "
                  f"{n_detected} false positive(s) flagged -- {'CLEAN' if n_detected == 0 else 'FALSE ALARM(S)'}")
        else:
            print(f"{name}: recall={r['recall']:.0%}  precision={r['precision']:.0%}  "
                  f"({n_detected} flagged / {n_gt} real erratic segments)")

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "erratic_driving_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
