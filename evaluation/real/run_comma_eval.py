"""
Pre-registered real-footage evaluation (see evaluation/real/PREREGISTRATION.md).

Fetch the footage first (not redistributed with CARAI):
    bash evaluation/real/fetch_comma.sh        # -> evaluation/real/data/comma/train.mp4, train.txt

Then:
    python3 evaluation/real/run_comma_eval.py [--max-frames N]

Labels are read ONLY after the estimator has produced every window.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from backend.dashcam.ego_speed import estimate_ego_speed, MarkingPrior  # noqa: E402

DATA = os.environ.get("CARAI_COMMA_DIR", os.path.join(ROOT, "evaluation", "real", "data", "comma"))
OUT = os.path.join(ROOT, "evaluation", "results")


def summarise(rows, label):
    ok = [r for r in rows if r["status"] == "OK"]
    out = {"segment": label, "n_windows": len(rows), "n_answered": len(ok),
           "answered_frac": len(ok) / len(rows) if rows else None}
    if ok:
        err = np.array([r["est"] - r["true"] for r in ok])
        ape = np.array([abs(r["est"] - r["true"]) / max(r["true"], 0.5) for r in ok])
        out.update({"mae_mps": float(np.mean(np.abs(err))), "median_abs_err_mps": float(np.median(np.abs(err))),
                    "median_ape": float(np.median(ape)), "p90_ape": float(np.percentile(ape, 90)),
                    "bias_mps": float(np.mean(err)),
                    "ci90_coverage": float(np.mean([r["ci"][0] <= r["true"] <= r["ci"][1] for r in ok])),
                    "mean_ci90_halfwidth_rel": float(np.mean([(r["ci"][1] - r["ci"][0]) / 2 / max(r["est"], 0.5) for r in ok]))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--prior", default="california_freeway", choices=["california_freeway", "us_federal"])
    args = ap.parse_args()
    video = os.path.join(DATA, "train.mp4")
    prior = MarkingPrior.preset(args.prior)
    t0 = time.time()
    res = estimate_ego_speed(video, prior, window_s=2.0, max_frames=args.max_frames)
    runtime = time.time() - t0
    # ---- labels are read only now ----
    labels = np.loadtxt(os.path.join(DATA, "train.txt"))
    fps = res["fps"]
    rows = []
    for w in res["windows"]:
        a, b = int(round(w["t_start_s"] * fps)), int(round(w["t_end_s"] * fps))
        true = float(labels[a:b].mean())
        rows.append({"t_start_s": w["t_start_s"], "frame0": a, "true": true, "status": w["status"],
                     "est": w["speed_mps"], "ci": w["ci90_mps"], "reasons": w["reasons"],
                     "segment": "freeway" if a < 9000 else "city"})
    summary = {"prior": res["prior"], "runtime_s": runtime, "n_frames": args.max_frames or len(labels),
               "all": summarise(rows, "all"),
               "freeway": summarise([r for r in rows if r["segment"] == "freeway"], "freeway (frames < 9000)"),
               "city": summarise([r for r in rows if r["segment"] == "city"], "city (frames >= 9000)")}
    # risk-coverage: rank answered windows by stated relative CI width (narrowest first)
    ok = sorted([r for r in rows if r["status"] == "OK"], key=lambda r: (r["ci"][1] - r["ci"][0]) / max(r["est"], 0.5))
    rc = []
    for k in range(1, len(ok) + 1):
        sub = ok[:k]
        rc.append({"coverage_of_all_windows": k / len(rows),
                   "median_ape": float(np.median([abs(r["est"] - r["true"]) / max(r["true"], 0.5) for r in sub]))})
    summary["risk_coverage"] = rc[:: max(1, len(rc) // 25)]
    reasons = {}
    for r in rows:
        for q in r["reasons"]:
            key = q.split("(")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
    summary["abstention_reasons"] = reasons
    os.makedirs(OUT, exist_ok=True)
    tag = f"comma_eval_{args.prior}"
    with open(os.path.join(OUT, tag + ".json"), "w") as f:
        json.dump({"summary": summary, "windows": rows}, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("all", "freeway", "city", "abstention_reasons", "runtime_s")}, indent=1))


if __name__ == "__main__":
    main()
