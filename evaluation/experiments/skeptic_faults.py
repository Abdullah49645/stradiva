"""
Fault-injection benchmark for the Skeptic (backend/verification/skeptic.py).

Question: when the reconstruction is wrong in a way the Reconstructor's own
fit cannot see, does the cross-examination catch it -- and how often does it
raise a false alarm on a correct reconstruction?

Scenes: two vehicles in the same lane at constant speed (no collision), on
simulator v2 with roadside and overhead cameras. For each scene the CLEAN
clip and several FAULTY variants are analysed by the unchanged production
pipeline:

  camera_bump          frames after mid-clip rotated 1.5 deg + shifted 6 px (pole knocked / PTZ nudge)
  wrong_standard_shape real road uses 6 m + 12 m dashes (duty 0.33) vs assumed 3 m + 9 m (0.25):
                       speed scale off by 1.5x
  misindexed_dash      algorithmic fault: dash indices beyond the middle dash shifted by one cycle
                       (what a mis-handled missing dash does)
  wrong_lateral_prior  real edge-to-centre distance 1.30 m vs assumed 1.85 m (lateral scale off 42 %)
  NEGATIVE CONTROLS (stated as undetectable by design -- reported, not hidden):
  same_duty_longer     real cycle 14.6 m (same 25 % duty) vs assumed 12 m: speed off 22 %
  wrong_fps_metadata   container claims 30 fps, scene rendered at 25 fps: speeds off 20 %

Reported per fault: share of runs the Skeptic disputes (per check), the
along-road speed error, and "silent failure" = |speed error| > 10 % with no
dispute and no rejection.
"""
import argparse
import json
import os
import sys
from unittest import mock

import cv2
import numpy as np

from common import CACHE_DIR, save_json, wilson_interval
from evaluation.sim.world import Scenario, CameraSpec, RoadSpec, VehicleSpec, Degradation, render_scenario

FAULTS = ["clean", "camera_bump", "wrong_standard_shape", "misindexed_dash", "wrong_lateral_prior",
          "same_duty_longer", "wrong_fps_metadata"]


def base_scene(i, road_kw=None, fps=30.0):
    rng = np.random.default_rng(700 + i)
    roadside = i % 2 == 1
    if roadside:
        side = 1.0 if i % 4 == 1 else -1.0
        off = float(rng.uniform(4, 8))
        cam = CameraSpec(height_m=float(rng.uniform(6, 9)), pitch_deg=float(rng.uniform(12, 18)),
                         roll_deg=float(rng.uniform(-3, 3)), yaw_deg=float(side * np.degrees(np.arctan(off / 55))),
                         y_m=-side * off, focal_px=float(rng.uniform(850, 1050)))
    else:
        cam = CameraSpec(height_m=float(rng.uniform(5.5, 8)), pitch_deg=float(rng.uniform(16, 24)),
                         roll_deg=float(rng.uniform(-3, 3)), yaw_deg=float(rng.uniform(-2, 2)), focal_px=float(rng.uniform(850, 1050)))
    rk = dict(topology="centerline", edge_half_width_m=1.85, dash_len_m=3.0, dash_gap_m=9.0,
              dash_phase_m=float(rng.uniform(0, 12)), wear_prob=0.05)
    rk.update(road_kw or {})
    lane = -rk["edge_half_width_m"] / 2.0
    v1 = float(rng.uniform(12, 24))
    v2 = v1 + float(rng.uniform(-3, -1))
    veh = [VehicleSpec("a", x0_m=12.0, v0_mps=v2, lane_y_m=lane, color_bgr=(40, 40, 190)),
           VehicleSpec("b", x0_m=12.0 + float(rng.uniform(25, 40)), v0_mps=v1, lane_y_m=lane, color_bgr=(40, 150, 40))]
    return Scenario(name=f"sk_{i:02d}", seed=700 + i, fps=fps, duration_s=4.0, camera=cam, road=RoadSpec(**rk),
                    vehicles=veh, degradation=Degradation(noise_std=3.0, blur_px=0.4))


def bump_video(src, dst, fps):
    cap = cv2.VideoCapture(src)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    h, w = frames[0].shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 1.5, 1.0)
    M[:, 2] += (6, 3)
    wr = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for k, f in enumerate(frames):
        wr.write(cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE) if k >= len(frames) // 2 else f)
    wr.release()


def relabel_fps(src, dst, fps_claim):
    cap = cv2.VideoCapture(src)
    w, h = int(cap.get(3)), int(cap.get(4))
    wr = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps_claim, (w, h))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        wr.write(f)
    wr.release()


def speed_errors(case, gt):
    """Relative error of each reported vehicle speed vs the true speed of the
    vehicle whose true image track is closest at the reference time."""
    out = []
    T = np.array([f["t_s"] for f in gt["frames"]])
    fps_true = gt["fps"]
    for sp in case.get("speeds") or []:
        if sp.get("withheld"):
            continue            # not reported as a measurement (bystander with implausible motion)
        veh = next((v for v in case.get("vehicles", []) if v["id"] == sp["vehicle_id"]), None)
        if veh is None:
            continue
        t_ref = sp["t_ref_s"]
        k = int(np.argmin(np.abs(np.array(veh["t_s"]) - t_ref)))
        uv = np.array(veh["uv"][k])
        # true frame index: the claimed fps may be wrong, so index by frame number
        fi = int(round(t_ref * case["input"]["fps"]))
        fi = min(max(fi, 0), len(gt["frames"]) - 1)
        best, bd = None, 1e9
        for vid, g in gt["frames"][fi]["vehicles"].items():
            d = np.hypot(*(np.array(g["rear_contact_uv"]) - uv))
            if d < bd:
                best, bd = g, d
        if best is not None and bd < 25:
            out.append(abs(sp["speed_mps"]) / best["speed_mps"] - 1.0)
    return out


def run(n_scenes):
    from backend.pipeline.analyze import analyze_video
    import backend.geometry.road_geometry as RG
    rows = []
    d = os.path.join(CACHE_DIR, "skeptic_faults")
    os.makedirs(d, exist_ok=True)
    for i in range(n_scenes):
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
                bump_video(vp, bp, fps)
                vp = bp
            if fault == "wrong_fps_metadata":
                fp = vp.replace(".mp4", "_claim30.mp4")
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
                    r = dict(r, k=k, a=float(coef[0]), b=float(coef[1]),
                             X_range=[float(min(k) * P), float(max(k) * P + L)])
                    return r
                with mock.patch.object(RG, "_fit_dash_index", faulty):
                    case = analyze_video(vp, seed=i)
            else:
                case = analyze_video(vp, seed=i)
            sk = case.get("skeptic") or {}
            errs = speed_errors(case, gt) if case.get("status") not in ("NOT_RECONSTRUCTABLE", "INVALID_INPUT") else []
            worst = float(max(np.abs(errs))) if errs else None
            rejected = case.get("status") in ("NOT_RECONSTRUCTABLE", "INVALID_INPUT", "MOVING_CAMERA")
            disputed = bool(sk.get("disputed"))
            blocking = case.get("status") == "DISPUTED"
            row = {"scene": i, "fault": fault, "status": case.get("status"), "rejected": rejected,
                   "disputed": disputed, "blocking_dispute": blocking, "disputed_checks": sk.get("disputed_checks"),
                   "speed_rel_errors": errs, "worst_speed_rel_error": worst,
                   "silent_failure": bool(worst is not None and worst > 0.10 and not disputed and not rejected),
                   "checks": {c["check"]: c["status"] for c in sk.get("checks", [])}}
            rows.append(row)
            print(f"scene {i} {fault:22s} status={row['status']:28s} disputed={row['disputed_checks']} "
                  f"worst_err={worst if worst is None else round(worst,3)} silent={row['silent_failure']}", flush=True)
    return rows


def summarise(rows):
    out = {}
    for fault in FAULTS:
        rs = [r for r in rows if r["fault"] == fault]
        if not rs:
            continue
        n = len(rs)
        k_d = sum(r["disputed"] or r["rejected"] for r in rs)
        k_s = sum(r["silent_failure"] for r in rs)
        per = {}
        for r in rs:
            for c, st in r["checks"].items():
                per.setdefault(c, 0)
                per[c] += st == "DISPUTE"
        errs = [r["worst_speed_rel_error"] for r in rs if r["worst_speed_rel_error"] is not None]
        k_b = sum(r["blocking_dispute"] or r["rejected"] for r in rs)
        out[fault] = {"n": n, "flagged_or_rejected": k_d, "flag_rate_ci90": wilson_interval(k_d, n),
                      "blocked_or_rejected": k_b, "block_rate_ci90": wilson_interval(k_b, n),
                      "silent_failures": k_s, "per_check_disputes": per,
                      "median_worst_speed_err": float(np.median(errs)) if errs else None}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    rows = run(a.n)
    summ = summarise(rows)
    print(json.dumps(summ, indent=1))
    save_json({"description": __doc__, "summary": summ, "rows": rows}, "skeptic_fault_injection.json")
