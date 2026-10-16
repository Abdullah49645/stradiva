"""
Pre-registered test (evaluation/real/PREREGISTRATION_LEAD.md): dashcam lead-vehicle range,
relative speed and TTC against the car's own radar, comma2k19 Example_1 segment 40.

  python3 run_comma2k19_lead.py --part dev     # t = 0-20 s, development (may be inspected and tuned)
  python3 run_comma2k19_lead.py --part test    # t = 20-60 s, run once on the frozen estimator

Writes evaluation/results/comma2k19_lead_{part}.json.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from backend.dashcam.lead_vehicle import LeadConfig, analyze_lead  # noqa: E402

SEG = os.environ.get("CARAI_COMMA2K19_SEG", os.path.join(ROOT, "evaluation", "real", "data", "comma2k19", "seg40"))
PARTS = {"dev": (0.0, 20.0), "test": (20.0, 60.0)}
CACHE = os.path.join(ROOT, "evaluation", "_sim_cache", "comma2k19_lead_estimate.json")


def radar_lead(frame_times):
    """Per frame: (forward distance, relative speed) of the nearest moving in-lane radar target."""
    t = np.load(os.path.join(SEG, "processed_log", "CAN", "radar", "t"))
    v = np.load(os.path.join(SEG, "processed_log", "CAN", "radar", "value"))
    ct = np.load(os.path.join(SEG, "processed_log", "CAN", "speed", "t"))
    cv = np.load(os.path.join(SEG, "processed_log", "CAN", "speed", "value")).ravel()
    out = []
    for tf in frame_times:
        m = (np.abs(t - tf) < 0.05) & (np.abs(v[:, 1]) < 1.6) & (v[:, 0] > 2) & np.isfinite(v[:, 0])
        ego = float(np.interp(tf, ct, cv))
        m &= np.abs(ego + v[:, 2]) > 2.0              # moving targets only (absolute speed > 2 m/s)
        if not m.any():
            out.append((np.nan, np.nan))
            continue
        k = np.argmin(v[m, 0])
        out.append((float(v[m, 0][k]), float(v[m, 2][k])))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=list(PARTS), required=True)
    ap.add_argument("--offset", type=float, default=None, help="camera-to-radar range offset (default: from the dev result)")
    a = ap.parse_args()
    ft = np.load(os.path.join(SEG, "global_pose", "frame_times"))
    fps = (len(ft) - 1) / (ft[-1] - ft[0])
    if os.path.exists(CACHE) and json.load(open(CACHE)).get("estimator_version") == ESTIMATOR_VERSION:
        est = json.load(open(CACHE))
    else:
        est = analyze_lead(os.path.join(SEG, "video.hevc"), fps=20.0, cfg=LeadConfig())
        est["estimator_version"] = ESTIMATOR_VERSION
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(est, open(CACHE, "w"))
    gt = radar_lead(ft)
    t0, t1 = PARTS[a.part]
    rows = []
    for w in est.get("windows", []):
        if not (t0 <= w["t_start_s"] < t1):
            continue
        i0, i1 = int(round(w["t_start_s"] * 20)), int(round(w["t_end_s"] * 20))
        g = gt[i0:i1]
        ok = np.isfinite(g[:, 0])
        row = {**w, "radar_range_m": float(np.median(g[ok, 0])) if ok.sum() >= 5 else None,
               "radar_rel_speed_mps": float(np.median(g[ok, 1])) if ok.sum() >= 5 else None}
        rr, vv = row["radar_range_m"], row["radar_rel_speed_mps"]
        row["radar_ttc_s"] = (rr / -vv) if (rr is not None and vv is not None and vv < -0.5) else None
        rows.append(row)
    if a.offset is not None:
        offset = a.offset
    elif a.part == "dev":
        d = [r["radar_range_m"] - r["range_m"] for r in rows if r["status"] == "OK" and r["radar_range_m"] is not None]
        offset = float(np.median(d)) if d else 0.0
    else:
        dev = json.load(open(os.path.join(ROOT, "evaluation", "results", "comma2k19_lead_dev.json")))
        offset = dev["summary"]["range_offset_m_from_dev"]
    ans = [r for r in rows if r["status"] == "OK" and r["radar_range_m"] is not None]
    summ = {"part": a.part, "t_range_s": [t0, t1], "n_windows": len(rows), "n_answered": sum(r["status"] == "OK" for r in rows),
            "n_answered_with_radar": len(ans), "range_offset_m_from_dev": offset, "camera_height_m": est.get("camera_height_m"),
            "estimator_version": ESTIMATOR_VERSION}
    if ans:
        ve = np.array([r["rel_speed_mps"] - r["radar_rel_speed_mps"] for r in ans])
        vc = [r["rel_speed_ci90_mps"][0] <= r["radar_rel_speed_mps"] <= r["rel_speed_ci90_mps"][1] for r in ans]
        re_ = np.array([(r["range_m"] + offset - r["radar_range_m"]) / r["radar_range_m"] for r in ans])
        rc = [r["range_ci90_m"][0] + offset <= r["radar_range_m"] <= r["range_ci90_m"][1] + offset for r in ans]
        summ.update({"rel_speed_median_abs_err_mps": float(np.median(np.abs(ve))), "rel_speed_ci90_coverage": float(np.mean(vc)),
                     "range_median_ape": float(np.median(np.abs(re_))), "range_ci90_coverage": float(np.mean(rc)),
                     "range_bias_rel": float(np.median(re_))})
        both = [r for r in ans if r.get("ttc_s") and r.get("radar_ttc_s")]
        summ["ttc_both_closing"] = len(both)
        summ["ttc_radar_inside_ci90"] = sum(r["ttc_ci90_s"][0] <= r["radar_ttc_s"] <= r["ttc_ci90_s"][1] for r in both)
        summ["ttc_carai_only"] = sum(1 for r in ans if r.get("ttc_s") and not r.get("radar_ttc_s"))
        summ["ttc_radar_only"] = sum(1 for r in ans if r.get("radar_ttc_s") and not r.get("ttc_s"))
    reasons = {}
    for r in rows:
        for x in r.get("reasons", []):
            reasons[x.split(" in ")[0]] = reasons.get(x.split(" in ")[0], 0) + 1
    summ["abstention_reasons"] = reasons
    print(json.dumps(summ, indent=1))
    for r in rows:
        print(f"{r['t_start_s']:5.1f} {r['status']:7s} Z={r.get('range_m', float('nan')):6.1f} radar={r['radar_range_m'] or float('nan'):6.1f}  "
              f"v={r.get('rel_speed_mps', float('nan')):+5.2f} radar={r['radar_rel_speed_mps'] if r['radar_rel_speed_mps'] is not None else float('nan'):+5.2f} "
              f"ttc={r.get('ttc_s')} radar_ttc={r.get('radar_ttc_s')} {r.get('reasons')}")
    with open(os.path.join(ROOT, "evaluation", "results", f"comma2k19_lead_{a.part}.json"), "w") as f:
        json.dump({"summary": summ, "windows": rows}, f, indent=1)


ESTIMATOR_VERSION = "lead-5"

if __name__ == "__main__":
    main()
