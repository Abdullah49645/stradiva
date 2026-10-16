"""
Tracking-layer diagnostics against simulator ground truth.

For each scenario: run the full pipeline, then match every ground-truth
vehicle to the pipeline's tracks frame by frame (box IoU) and report

  coverage        fraction of frames with the vehicle's ground contact in view
                  in which a MEASURED track record overlaps it (IoU >= 0.3)
  n_tracks        distinct track ids that covered it (1 = no identity break)
  contact_err_px  median / 90th percentile |measured contact - true contact|
  pair_correct    did the pipeline choose the true (follower, lead) tracks

This isolates the detection/tracking layer from the reasoning layer
(the oracle modes of collision_suite.py do the converse).

  python3 track_diagnostics.py --split dev --cases 1 3 4
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE_DIR, save_json  # noqa: E402
from collision_suite import sample_scenario  # noqa: E402

from evaluation.sim.world import render_scenario  # noqa: E402


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(1e-9, aw * ah + bw * bh - inter)


def diagnose(split, i):
    from backend.pipeline.analyze import analyze_video
    out_dir = os.path.join(CACHE_DIR, "collision_suite", split)
    sc, meta = sample_scenario(split, i)
    gpath = os.path.join(out_dir, sc.name + "_ground_truth.json")
    if not os.path.exists(gpath):
        render_scenario(sc, out_dir)
    gt = json.load(open(gpath))
    case, internals = analyze_video(os.path.join(out_dir, sc.name + ".mp4"), seed=0, return_internals=True)
    tracks = internals["tracks"]
    by_frame = {}
    for tr in tracks:
        for r in tr["records"]:
            if r.get("bbox") and r["status"] in ("MEASURED", "MEASURED_OCCLUDING"):
                by_frame.setdefault(r["frame"], []).append((tr["id"], r))
    dirs = {v.vid: v.direction for v in sc.vehicles}
    res = {"case": sc.name, "family": meta.get("family"), "status": case.get("status"),
           "stages": [(s["stage"], s["status"], s["detail"]) for s in case.get("stages", [])], "vehicles": {}}
    owner = {}
    for vid in dirs:
        n_vis = n_cov = 0
        ids, errs = {}, []
        for f in gt["frames"]:
            g = f["vehicles"][vid]
            if g["bbox"] is None:
                continue
            uv = np.array(g["rear_contact_uv" if dirs[vid] > 0 else "front_contact_uv"], float)
            if not (np.isfinite(uv).all() and 0 < uv[0] < sc.camera.image_w and 0 < uv[1] < sc.camera.image_h):
                continue
            n_vis += 1
            best = max(((iou(r["bbox"], g["bbox"]), tid, r) for tid, r in by_frame.get(f["frame"], [])),
                       default=(0, None, None), key=lambda z: z[0])
            if best[0] >= 0.3:
                n_cov += 1
                ids[best[1]] = ids.get(best[1], 0) + 1
                c = best[2].get("contact_uv")
                if c is not None:
                    errs.append(float(np.hypot(*(np.array(c) - uv))))
        owner[vid] = max(ids, key=ids.get) if ids else None
        res["vehicles"][vid] = {"frames_visible": n_vis, "coverage": n_cov / max(1, n_vis),
                                "track_ids": ids, "n_tracks": len(ids),
                                "contact_err_px_median": float(np.median(errs)) if errs else None,
                                "contact_err_px_p90": float(np.percentile(errs, 90)) if errs else None}
    pair = internals.get("pair")
    if pair:
        chosen = (pair[0]["track"]["id"], pair[1]["track"]["id"])
        # a stitched vehicle may carry several ids; compare with the dominant one
        res["pair_chosen"] = chosen
        res["pair_correct"] = bool(owner.get("follow") in _ids(pair[0]) and owner.get("lead") in _ids(pair[1]))
    else:
        res["pair_chosen"] = None
        res["pair_correct"] = False
    res["skeptic_disputed"] = (case.get("skeptic") or {}).get("disputed_checks")
    res["kinematics"] = next((c for c in (case.get("skeptic") or {}).get("checks", []) if c["check"] == "kinematics"), None)
    return res


def _ids(v):
    return set(v["track"].get("stitched_from") or [v["track"]["id"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--cases", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--variant", default=None, choices=[None, "no_first_half"],
                    help="ablation: background refinement without the first-half fallback")
    a = ap.parse_args()
    if a.variant == "no_first_half":
        import functools
        import backend.pipeline.analyze as _an
        _an.refine_background = functools.partial(_an.refine_background, use_first_half_where_changed=False)
    rows = []
    for i in a.cases:
        r = diagnose(a.split, i)
        rows.append(r)
        v = r["vehicles"]
        print(f"{r['case']} {r['family']:16s} {r['status']:28s} pair_ok={r['pair_correct']!s:5s} "
              + "  ".join(f"{k}: cov {d['coverage']:.2f} ids {d['n_tracks']} err {d['contact_err_px_median'] or float('nan'):.1f}px"
                          for k, d in v.items()), flush=True)
    if a.out:
        save_json(rows, a.out)
