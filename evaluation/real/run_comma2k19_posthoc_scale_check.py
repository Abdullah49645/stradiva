"""
POST-HOC (after the pre-registered holdout failed; see PREREGISTRATION.md, Deviations, and
docs/EVALUATION.md section 5): the same comma2k19 segment with the lane-width x focal-length scale
cross-check enabled.

  caltrans_freeway_48ft   the pre-registered prior; the cross-check should refuse it
  urban_24ft              the 10 ft + 14 ft city broken-line pattern (cycle 24 ft = 7.32 m) that the
                          cross-check's implied cycle points to -- chosen AFTER seeing the data, so a
                          demonstration, not a validation

Focal length 910 px: comma2k19's documented forward-camera intrinsics (EON, 1164x874). Frame rate
20 fps from the segment's sensor timestamps (the container says 25).

  python3 run_comma2k19_posthoc_scale_check.py   -> evaluation/results/comma2k19_holdout_posthoc_scale_check.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from backend.dashcam.ego_speed import EgoSpeedConfig, MarkingPrior, estimate_ego_speed  # noqa: E402

SEG = os.environ.get("CARAI_COMMA2K19_SEG", os.path.join(ROOT, "evaluation", "real", "data", "comma2k19", "seg40"))
PRIORS = {"caltrans_freeway_48ft": MarkingPrior.preset("california_freeway"),
          "urban_24ft": MarkingPrior("city broken line 10 ft + 14 ft", 24 * 0.3048)}


def main():
    ft = np.load(os.path.join(SEG, "global_pose", "frame_times"))
    ct = np.load(os.path.join(SEG, "processed_log", "CAN", "speed", "t"))
    cv = np.load(os.path.join(SEG, "processed_log", "CAN", "speed", "value")).ravel()
    v_frame = np.interp(ft, ct, cv)
    out = {}
    for name, prior in PRIORS.items():
        res = estimate_ego_speed(os.path.join(SEG, "video.hevc"), prior, window_s=2.0, fps_override=20.0,
                                 cfg=EgoSpeedConfig(focal_px=910.0, lane_width_prior_m=3.6))
        rows = []
        for w in res["windows"]:
            a, b = int(round(w["t_start_s"] * 20)), int(round(w["t_end_s"] * 20))
            xc = w["diagnostics"].get("scale_cross_check") or {}
            rows.append({"frame0": a, "true": float(v_frame[a:b].mean()), "status": w["status"], "est": w["speed_mps"],
                         "ci": w["ci90_mps"], "implied_cycle_m": xc.get("implied_cycle_m"), "xc": xc.get("status"),
                         "reasons": w["reasons"]})
        ok = [r for r in rows if r["status"] == "OK"]
        implied = [r["implied_cycle_m"] for r in rows if r["implied_cycle_m"]]
        summ = {"answered": len(ok), "n": len(rows),
                "implied_cycle_median_m": float(np.median(implied)) if implied else None,
                "n_rejected_by_scale_check": sum("inconsistent with lane geometry" in " ".join(r["reasons"]) for r in rows)}
        if ok:
            ape = np.array([abs(r["est"] - r["true"]) / r["true"] for r in ok])
            summ.update({"median_ape": float(np.median(ape)), "max_ape": float(ape.max()),
                         "coverage": float(np.mean([r["ci"][0] <= r["true"] <= r["ci"][1] for r in ok]))})
        out[name] = {"summary": summ, "windows": rows}
        print(name, json.dumps(summ))
    with open(os.path.join(ROOT, "evaluation", "results", "comma2k19_holdout_posthoc_scale_check.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
