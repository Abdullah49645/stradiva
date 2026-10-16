"""
The investigator (backend/investigator) on the Skeptic's fault-injection clips
(evaluation/experiments/skeptic_faults.py: same scenes, same faults, same
cached renders).

Question: when the Skeptic blocks a reconstruction, does the investigator
recover a CORRECT answer, and does it ever turn a refusal into a wrong one?

Per clip, for the base pipeline and for the investigator's final case:
  answered     speeds reported and no blocking dispute / rejection
  worst error  largest |relative speed error| of a reported (non-withheld) speed
  covered      every reported speed's 90 % interval contains the true speed
  harmful      answered, no flag, and worst error > 10 %

  python3 investigator_faults.py --n 8   -> evaluation/results/investigator_faults.json
"""
import argparse
import json
import os
from unittest import mock

import numpy as np

from common import CACHE_DIR, save_json, wilson_interval
from evaluation.sim.world import render_scenario
from skeptic_faults import FAULTS, base_scene, bump_video, relabel_fps, speed_errors


def coverage(case, gt):
    T = gt["frames"]
    ok = []
    for sp in case.get("speeds") or []:
        if sp.get("withheld") or not sp.get("ci90_mps"):
            continue
        veh = next((v for v in case.get("vehicles", []) if v["id"] == sp["vehicle_id"]), None)
        if veh is None:
            continue
        k = int(np.argmin(np.abs(np.array(veh["t_s"]) - sp["t_ref_s"])))
        uv = np.array(veh["uv"][k])
        fi = min(max(int(round(sp["t_ref_s"] * case["input"]["fps"])), 0), len(T) - 1)
        best, bd = None, 1e9
        for g in T[fi]["vehicles"].values():
            d = np.hypot(*(np.array(g["rear_contact_uv"]) - uv))
            if d < bd:
                best, bd = g, d
        if best is not None and bd < 25:
            lo, hi = sp["ci90_mps"]
            ok.append(lo <= abs(best["speed_mps"]) <= hi or lo <= -abs(best["speed_mps"]) <= hi)
    return ok


def assess(case, gt):
    rejected = case.get("status") in ("NOT_RECONSTRUCTABLE", "INVALID_INPUT", "MOVING_CAMERA")
    blocked = case.get("status") in ("DISPUTED", "STANDARD_DEPENDENT") or rejected
    errs = speed_errors(case, gt) if not rejected else []
    worst = float(max(np.abs(errs))) if errs else None
    cov = coverage(case, gt) if not rejected else []
    answered = (not blocked) and worst is not None
    flagged = bool((case.get("skeptic") or {}).get("disputed"))
    return {"status": case.get("status"), "answered": answered, "worst_err": worst,
            "covered": bool(cov) and all(cov), "harmful": bool(answered and worst > 0.10 and not flagged)}


def run(start, n):
    import backend.geometry.road_geometry as RG
    from backend.investigator.investigator import investigate
    d = os.path.join(CACHE_DIR, "skeptic_faults")
    os.makedirs(d, exist_ok=True)
    rows = []
    for i in range(start, start + n):
        for fault in FAULTS:
            road_kw, fps = None, 30.0
            if fault == "wrong_standard_shape":
                road_kw = {"dash_len_m": 6.0, "dash_gap_m": 12.0}
            elif fault == "wrong_lateral_prior":
                road_kw = {"edge_half_width_m": 1.30}
            elif fault == "same_duty_longer":
                road_kw = {"dash_len_m": 3.66, "dash_gap_m": 10.97}
            elif fault == "wrong_fps_metadata":
                fps = 25.0
            sc = base_scene(i, road_kw, fps)
            sc.name = f"sk_{i:02d}_{fault}"
            vp, gp, gt = render_scenario(sc, d)
            if fault == "camera_bump":
                bp = vp.replace(".mp4", "_bumped.mp4")
                if not os.path.exists(bp):
                    bump_video(vp, bp, fps)
                vp = bp
            if fault == "wrong_fps_metadata":
                fp = vp.replace(".mp4", "_claim30.mp4")
                if not os.path.exists(fp):
                    relabel_fps(vp, fp, 30.0)
                vp = fp
            if fault == "misindexed_dash":
                orig = RG._fit_dash_index

                def faulty(dash_runs, prior, cfg, _orig=orig):
                    r = _orig(dash_runs, prior, cfg)
                    if r is None:
                        return r
                    k = list(r["k"])
                    mid = len(k) // 2
                    k = k[:mid] + [kk + 1 for kk in k[mid:]]
                    P, L = prior.cycle_m, prior.dash_len_m
                    X = np.array(k) * P + L / 2.0
                    cen = np.array(r["centres"])
                    w_ = np.sqrt(1.0 / cen ** 4)
                    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(X), X]) * w_[:, None], cen * w_, rcond=None)
                    return dict(r, k=k, a=float(coef[0]), b=float(coef[1]), X_range=[float(min(k) * P), float(max(k) * P + L)])
                with mock.patch.object(RG, "_fit_dash_index", faulty):
                    case, base_case = investigate(vp, seed=i, return_base=True)
            else:
                case, base_case = investigate(vp, seed=i, return_base=True)
            inv = case["investigation"]
            final = assess(case, gt)
            row = {"scene": i, "fault": fault, "final": final, "base": assess(base_case, gt),
                   "changed": inv["changed"], "trace": inv["trace"], "requests": inv["requests"]}
            rows.append(row)
            print(f"scene {i} {fault:22s} base={inv['base']['status']:26s} final={final['status']:26s} "
                  f"answered={final['answered']} err={final['worst_err'] if final['worst_err'] is None else round(final['worst_err'], 3)} "
                  f"covered={final['covered']} harmful={final['harmful']}", flush=True)
    return rows


def summarise(rows):
    out = {}
    for fault in FAULTS:
        rs = [r for r in rows if r["fault"] == fault]
        if not rs:
            continue
        n = len(rs)
        b_ans = sum(r["base"]["answered"] for r in rs)
        ans = [r for r in rs if r["final"]["answered"]]
        good = [r for r in ans if r["final"]["worst_err"] <= 0.10 or r["final"]["covered"]]
        recovered = [r for r in rs if r["changed"] and r["final"]["answered"]]
        out[fault] = {"n": n, "base_answered": b_ans, "final_answered": len(ans),
                      "final_answered_rate_ci90": wilson_interval(len(ans), n),
                      "recovered_by_investigator": len(recovered),
                      "recovered_correct": sum(r in good for r in recovered),
                      "final_answered_correct_or_covered": len(good),
                      "harmful": sum(r["final"]["harmful"] for r in rs),
                      "base_harmful": sum(r["base"]["harmful"] for r in rs),
                      "made_harmful_by_investigator": sum(r["final"]["harmful"] and not r["base"]["harmful"] for r in rs),
                      "median_worst_err_answered": float(np.median([r["final"]["worst_err"] for r in ans])) if ans else None}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0, help="scenes 0-7: development (the remedies were built on them); "
                                                           "8-15: held out")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="investigator_faults.json")
    ap.add_argument("--merge", nargs="+", help="merge result files (written in parallel) into --out instead of running")
    a = ap.parse_args()
    if a.merge:
        parts = [json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", m))) for m in a.merge]
        rows = [r for p in parts for r in p["rows"]]
        sc = [min(p["scenes"][0] for p in parts), max(p["scenes"][1] for p in parts)]
        save_json({"description": __doc__, "scenes": sc, "merged_from": a.merge, "summary": summarise(rows), "rows": rows}, a.out)
        print(json.dumps(summarise(rows), indent=1))
        raise SystemExit
    rows = run(a.start, a.n)
    summ = summarise(rows)
    print(json.dumps(summ, indent=1))
    save_json({"description": __doc__, "scenes": [a.start, a.start + a.n - 1], "summary": summ, "rows": rows}, a.out)
