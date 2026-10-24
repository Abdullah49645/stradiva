"""
Independent holdout (see evaluation/real/PREREGISTRATION_HOLDOUT.md): comma2k19 example segment,
CAN speed as ground truth. Fetch:  git clone https://github.com/commaai/comma2k19  (MIT) and point
STRADIVA_COMMA2K19_SEG at  Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from backend.dashcam.ego_speed import estimate_ego_speed, MarkingPrior  # noqa: E402

SEG = os.environ.get("STRADIVA_COMMA2K19_SEG", os.path.join(ROOT, "evaluation", "real", "data", "comma2k19", "seg40"))


def run(fps_override):
    res = estimate_ego_speed(os.path.join(SEG, "video.hevc"), MarkingPrior.preset("california_freeway"), window_s=2.0,
                             fps_override=fps_override)
    ft = np.load(os.path.join(SEG, "global_pose", "frame_times"))
    ct = np.load(os.path.join(SEG, "processed_log", "CAN", "speed", "t"))
    cv = np.load(os.path.join(SEG, "processed_log", "CAN", "speed", "value")).ravel()
    v_frame = np.interp(ft, ct, cv)
    true_fps = (len(ft) - 1) / (ft[-1] - ft[0])
    rows = []
    for w in res["windows"]:
        # window boundaries in FRAME indices are what the estimator actually used
        a = int(round(w["t_start_s"] * res["fps"]))
        b = int(round(w["t_end_s"] * res["fps"]))
        true = float(v_frame[a:b].mean())
        rows.append({"frame0": a, "true": true, "status": w["status"], "est": w["speed_mps"], "ci": w["ci90_mps"],
                     "reasons": w["reasons"]})
    ok = [r for r in rows if r["status"] == "OK"]
    summ = {"fps_used": res["fps"], "fps_source": res["fps_source"], "fps_container": res["fps_container"],
            "fps_from_sensor_timestamps": true_fps, "n_windows": len(rows), "n_answered": len(ok)}
    if ok:
        ape = np.array([abs(r["est"] - r["true"]) / r["true"] for r in ok])
        summ.update({"median_ape": float(np.median(ape)), "max_ape": float(ape.max()),
                     "ci90_coverage": float(np.mean([r["ci"][0] <= r["true"] <= r["ci"][1] for r in ok])),
                     "catastrophic_windows_gt25pct": int(np.sum(ape > 0.25)),
                     "speed_range_true_mps": [float(min(r["true"] for r in rows)), float(max(r["true"] for r in rows))]})
    return summ, rows


if __name__ == "__main__":
    out = {}
    for label, fo in (("fps_from_sensor_timestamps", 20.0), ("fps_from_container_unverified", None)):
        s, rows = run(fo)
        out[label] = {"summary": s, "windows": rows}
        print(label, json.dumps(s, indent=1))
    with open(os.path.join(ROOT, "evaluation", "results", "comma2k19_holdout.json"), "w") as f:
        json.dump(out, f, indent=1)
