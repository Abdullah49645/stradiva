"""
Collision evaluation suite: randomized rear-end scenarios on simulator v2,
an INDEPENDENT ground-truth counterfactual, and the full production pipeline
(backend/pipeline/analyze.py) with no access to ground truth.

Splits (fixed before any test-split result was looked at):
  dev    seeds 1000+i  standard ranges   -- the ONLY split used to set thresholds/parameters
  test   seeds 5000+i  standard ranges   -- held out, i.i.d. with dev
  shift  seeds 9000+i  harder ranges     -- held out, distribution shift (more roll/yaw, worse
                                            image quality, larger marking-prior mismatch,
                                            more oncoming-traffic scenes)

Ground truth (computed here, not by CARAI code):
  * hazard onset = the instant the lead's scripted braking begins;
  * counterfactual: follower follows its true (free) motion until onset +
    reaction time, then brakes at a constant a to rest; lead follows its
    true scripted motion (as if it had not been hit);
  * same reaction/deceleration grid and envelope as CARAI ->
    AVOIDABLE / NOT_AVOIDABLE / ASSUMPTION_DEPENDENT, and t_r*(a).

Run:  python3 collision_suite.py --split dev --n 30 --workers 2
"""
import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np

from common import ROOT, CACHE_DIR, RESULTS_DIR  # noqa: F401
from evaluation.sim.world import (Scenario, CameraSpec, RoadSpec, VehicleSpec, MotionSegment, Degradation,
                                  render_scenario, simulate_vehicle)

REACTION_GRID = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
DECEL_GRID = [4.0, 5.0, 6.0, 7.0, 8.0]
ENV_R, ENV_A = (0.75, 2.0), (5.0, 8.0)
ASSUMED_CYCLE = 12.0
ASSUMED_EDGE = 1.85


# ----------------------------------------------------------------- sampler --

def sample_scenario(split, i):
    # test2: a second held-out split with the test distribution, added (and committed) AFTER the eval-freeze
    # run and BEFORE any post-freeze change is evaluated on it -- see docs/EVALUATION.md section 6
    base = {"dev": 1000, "test": 5000, "shift": 9000, "test2": 13000}[split]
    seed = base + i
    rng = np.random.default_rng(seed)
    hard = split == "shift"
    facing = rng.random() < (0.45 if hard else 0.3)
    roadside = rng.random() < 0.45
    if roadside:
        # pole-mounted camera beside the road, aimed along it (the commonest real CCTV setup)
        side = float(rng.choice([-1.0, 1.0]))
        off = float(rng.uniform(4.0, 9.0))
        aim = float(np.degrees(np.arctan(off / rng.uniform(40, 70))))
        cam = CameraSpec(height_m=float(rng.uniform(6.0, 10.0)), pitch_deg=float(rng.uniform(12, 20)),
                         roll_deg=float(rng.uniform(-10, 10) if hard else rng.uniform(-4, 4)),
                         yaw_deg=float(side * aim + rng.uniform(-2, 2)), y_m=-side * off,
                         focal_px=float(rng.uniform(800, 1100)))
    else:
        cam = CameraSpec(
            height_m=float(rng.uniform(5.0, 9.0)), pitch_deg=float(rng.uniform(15, 26)),
            roll_deg=float(rng.uniform(-10, 10) if hard else rng.uniform(-5, 5)),
            yaw_deg=float(rng.uniform(-8, 8) if hard else rng.uniform(-3, 3)),
            focal_px=float(rng.uniform(800, 1100)))
    prior_sd = 0.05 if hard else 0.03
    cyc = ASSUMED_CYCLE * (1 + rng.normal(0, prior_sd))
    road = RoadSpec(topology="centerline", edge_half_width_m=float(ASSUMED_EDGE * (1 + rng.normal(0, 0.03))),
                    dash_len_m=float(cyc * 0.25), dash_gap_m=float(cyc * 0.75),
                    dash_phase_m=float(rng.uniform(0, cyc)), wear_prob=float(rng.uniform(0, 0.25 if hard else 0.12)),
                    paint_level=int(rng.uniform(150 if hard else 185, 240)))
    deg = Degradation(noise_std=float(rng.uniform(2, 8 if hard else 5)),
                      blur_px=float(rng.uniform(0, 1.4 if hard else 0.8)),
                      fog_visibility_m=float(rng.uniform(150, 300)) if rng.random() < (0.3 if hard else 0.1) else 0.0,
                      brightness=float(rng.uniform(0.55, 1.0) if hard else rng.uniform(0.8, 1.05)))
    lane_y = -road.edge_half_width_m / 2.0 + float(rng.uniform(-0.15, 0.15))
    v_f = float(rng.uniform(12, 27))
    v_l = float(np.clip(v_f + rng.uniform(-4, 1.5), 6, 30))
    gap0 = float(rng.uniform(14, 38))
    t_b = float(rng.uniform(0.6, 1.8))
    a_l = float(rng.uniform(4.5, 8.0))
    brakes = rng.random() > 0.15
    r_true = float(rng.uniform(0.5, 2.6))
    a_f = float(rng.uniform(3.0, 8.0))
    L_f, L_l = float(rng.uniform(4.2, 4.9)), float(rng.uniform(4.2, 4.9))
    palette = [(30, 40, 190), (40, 150, 40), (160, 120, 40), (90, 90, 95), (200, 200, 205), (40, 40, 45), (20, 140, 200)]
    cf, cl = rng.choice(len(palette), 2, replace=False)
    d = -1 if facing else 1
    if not facing:
        x_f0 = float(rng.uniform(10, 18))
        x_l0 = x_f0 + gap0 + (L_f + L_l) / 2.0
    else:
        x_l0 = float(rng.uniform(60, 85))          # far enough that the event happens in view
        x_f0 = x_l0 + gap0 + (L_f + L_l) / 2.0
    lead = VehicleSpec("lead", x0_m=x_l0, v0_mps=v_l, lane_y_m=lane_y, length_m=L_l, color_bgr=palette[cl],
                       segments=[MotionSegment(0, 0.0), MotionSegment(t_b, -a_l)], direction=d)
    fsegs = [MotionSegment(0, 0.0)]
    if brakes:
        fsegs.append(MotionSegment(t_b + r_true, -a_f))
    fol = VehicleSpec("follow", x0_m=x_f0, v0_mps=v_f, lane_y_m=lane_y, length_m=L_f, color_bgr=palette[cf],
                      segments=fsegs, direction=d)
    sc = Scenario(name=f"{split}_{i:03d}", seed=seed, fps=30.0, duration_s=6.0, camera=cam, road=road,
                  vehicles=[lead, fol], degradation=deg)
    meta = {"family": ("roadside_" if roadside else "overhead_") + ("facing" if facing else "behind"),
            "hazard_onset_s": t_b, "true_cycle_m": cyc,
            "follower_brakes": bool(brakes), "follower_reaction_true_s": r_true if brakes else None}
    return sc, meta


# ------------------------------------------------------- ground truth (indep.) --

def _free(spec, t):
    return simulate_vehicle(spec, np.atleast_1d(t))


def gt_counterfactual(sc: Scenario, meta, t_contact):
    lead = next(v for v in sc.vehicles if v.vid == "lead")
    fol = next(v for v in sc.vehicles if v.vid == "follow")
    d = lead.direction
    t_h = meta["hazard_onset_s"]
    tt = np.arange(0.0, sc.duration_s + 10.0, 0.004)
    xl, vl, _ = _free(lead, tt)
    xf, vf, _ = _free(fol, tt)
    half = (lead.length_m + fol.length_m) / 2.0

    def avoided(r, a):
        tr = t_h + r
        if t_contact is not None and tr >= t_contact:
            return False
        k = int(np.searchsorted(tt, tr))
        x0, v0 = xf[k], vf[k]
        tau = tt[k:] - tt[k]
        ts = v0 / a
        s = np.where(tau < ts, v0 * tau - 0.5 * a * tau ** 2, v0 * ts - 0.5 * a * ts ** 2)
        gap = d * (xl[k:] - (x0 + d * s)) - half
        # before the reaction instant the follower moves as it really did, and the real contact
        # had not happened yet (tr < t_contact), so only the braking phase can produce contact
        return bool(np.min(gap) > 0)

    cells = [[avoided(r, a) for r in REACTION_GRID] for a in DECEL_GRID]
    env = [cells[i][j] for i, a in enumerate(DECEL_GRID) for j, r in enumerate(REACTION_GRID)
           if ENV_A[0] <= a <= ENV_A[1] and ENV_R[0] <= r <= ENV_R[1]]
    verdict = "AVOIDABLE" if all(env) else ("NOT_AVOIDABLE" if not any(env) else "ASSUMPTION_DEPENDENT")
    crit = {}
    for a in DECEL_GRID:
        lo, hi = 0.0, 4.0
        if not avoided(lo, a):
            crit[f"{a:g}"] = 0.0
            continue
        if avoided(hi, a):
            crit[f"{a:g}"] = hi
            continue
        while hi - lo > 0.005:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if avoided(mid, a) else (lo, mid)
        crit[f"{a:g}"] = lo
    k_h = int(np.searchsorted(tt, t_h))
    return {"verdict": verdict, "cells": cells, "critical_reaction_s": crit,
            "follower_speed_at_onset_mps": float(vf[k_h]), "lead_speed_at_onset_mps": float(vl[k_h])}


# ------------------------------------------------------------------ scoring --

def gt_observable(gt, sc, t_event):
    """Both vehicles' visible ground contacts are inside the image at the event time."""
    if t_event is None:
        return True
    k = min(len(gt["frames"]) - 1, int(round(t_event * gt["fps"])))
    f = gt["frames"][k]["vehicles"]
    ok = True
    for v in sc.vehicles:
        uv = f[v.vid]["rear_contact_uv" if v.direction > 0 else "front_contact_uv"]
        ok &= bool(np.isfinite(uv).all() and 0 < uv[0] < sc.camera.image_w and 0 < uv[1] < sc.camera.image_h)
    return ok


def event_resolution(gt, sc, t_event):
    """Ground truth only (for stratifying results, never used by CARAI):
    along-road metres covered by one image row at the follower's visible
    ground contact at the event time (collision, else closest approach),
    and the follower's distance from the camera there."""
    fr = gt["frames"]
    if t_event is None:     # closest approach of the pair
        d = [abs(f["vehicles"]["lead"]["x_m"] - f["vehicles"]["follow"]["x_m"]) for f in fr]
        k = int(np.argmin(d))
    else:
        k = min(len(fr) - 1, int(round(t_event * gt["fps"])))
    fv = fr[k]["vehicles"]["follow"]
    uv = np.array(fv["rear_contact_uv" if sc.vehicles[[v.vid for v in sc.vehicles].index("follow")].direction > 0
                     else "front_contact_uv"], float)
    if not np.isfinite(uv).all():
        return None
    Hinv = np.linalg.inv(np.array(gt["true_world_to_image_H_at_t0"]))
    def road(p):
        q = Hinv @ np.array([p[0], p[1], 1.0])
        return q[:2] / q[2]
    m_per_row = float(np.linalg.norm(road(uv + [0, 0.5]) - road(uv - [0, 0.5])))
    return {"m_per_px_along_road": m_per_row, "range_m": float(abs(fv["x_m"] - sc.camera.x_m))}


NO_VERDICT = {"NOT_RECONSTRUCTABLE", "INSUFFICIENT_EVIDENCE", "NO_COLLISION_ANALYSED", "NO_PAIR_RECONSTRUCTED"}
CONTACT_TIME_TOL_S = 0.3     # a reported contact more than this far from the true one is a different event


def outcome(row):
    """Scoring categories:
    correct / correct_abstention / abstained / overconfident / catastrophic /
    false_collision (contact reported where none happened) /
    missed_collision (a visible collision reported as "no contact found") /
    missed_unobservable (an off-screen collision reported as "no contact found") /
    wrong_contact_time (a verdict whose reported contact is > CONTACT_TIME_TOL_S from the true one)."""
    v = row.get("carai_verdict")
    if row.get("status") == "RECONSTRUCTED_NO_INTERACTION" and v == "NO_COLLISION_ANALYSED":
        v = "NO_PAIR_RECONSTRUCTED"     # rows written before this verdict existed: same pipeline state
    gtv = row["gt"]["verdict"]
    if row.get("gt_collision_t_s") is None:                 # nothing happened
        if v in ("AVOIDABLE", "NOT_AVOIDABLE", "ASSUMPTION_DEPENDENT"):
            return "false_collision"
        return "correct" if v == "NO_COLLISION_ANALYSED" else "abstained"
    if not row.get("gt_observable", True):                  # the event is off-screen
        if v == "NO_COLLISION_ANALYSED":
            return "missed_unobservable"                     # said "no contact found" -- true of the footage, still a miss
        return "correct_abstention" if v in NO_VERDICT else ("catastrophic" if v in ("AVOIDABLE", "NOT_AVOIDABLE") else "qualified_unobservable")
    if v == "NO_COLLISION_ANALYSED":
        return "missed_collision"                            # a visible collision reported as no contact
    tc = row.get("carai_contact_t_s")
    if v in ("AVOIDABLE", "NOT_AVOIDABLE", "ASSUMPTION_DEPENDENT") and tc is not None \
            and abs(tc - row["gt_collision_t_s"]) > CONTACT_TIME_TOL_S:
        return "wrong_contact_time"                          # a verdict about a contact that is not the real one
    if v in NO_VERDICT or v is None:
        return "abstained"
    return classify(v, gtv)


def classify(carai_verdict, gt_verdict):
    definite = {"AVOIDABLE", "NOT_AVOIDABLE"}
    if carai_verdict == gt_verdict:
        return "correct"
    if carai_verdict in definite and gt_verdict in definite:
        return "catastrophic"          # opposite definite verdicts
    if carai_verdict in definite and gt_verdict == "ASSUMPTION_DEPENDENT":
        return "overconfident"         # claims robustness the physics does not have
    return "abstained"                 # CARAI qualified/withheld where GT was definite


def oracle_tracks(gt, sc, occluded, rng, sigma_px=0.7):
    """Tracks built from GROUND-TRUTH ground-contact points (+ pixel noise) in
    the pipeline's own record format. 'occluded': the vehicle farther from the camera is unmeasured in
    frames where the nearer vehicle's true box overlaps it (what a camera sees)."""
    vids = [v.vid for v in sc.vehicles]
    dirs = {v.vid: v.direction for v in sc.vehicles}
    tracks = []
    for n, vid in enumerate(vids):
        recs = []
        for f in gt["frames"]:
            g = f["vehicles"][vid]
            if g["bbox"] is None:
                continue
            uv = np.array(g["rear_contact_uv"] if dirs[vid] > 0 else g["front_contact_uv"]) + rng.normal(0, sigma_px, 2)
            x, y, bw, bh = g["bbox"]
            if not (0 < uv[0] < sc.camera.image_w - 2 and 0 < uv[1] < sc.camera.image_h - 2):
                continue
            status = "MEASURED"
            # the vehicle FARTHER from the camera is the one that gets hidden: the lead when traffic recedes
            # (direction +1), the follower when it approaches (direction -1)
            far_vid = "lead" if dirs["lead"] > 0 else "follow"
            if occluded and vid == far_vid:
                o = f["vehicles"]["follow" if far_vid == "lead" else "lead"]["bbox"]
                if o is not None:
                    ix = max(0, min(x + bw, o[0] + o[2]) - max(x, o[0]))
                    iy = max(0, min(y + bh, o[1] + o[3]) - max(y, o[1]))
                    if ix * iy > 0:
                        status = "OCCLUDED"
            recs.append({"frame": f["frame"], "bbox": [float(v) for v in g["bbox"]], "status": status,
                         "contact_uv": [float(uv[0]), float(uv[1])],
                         "contact_span": [float(x), float(x + bw)] if status == "MEASURED" else None})
        tracks.append({"id": n, "records": recs, "n_measured": sum(r["status"] == "MEASURED" for r in recs)})
    return tracks


def run_oracle(split, i, occluded):
    from backend.pipeline.analyze import analyze_from_tracks, ingest, PIPELINE_VERSION
    from backend.geometry.road_geometry import RoadPrior, GeometryConfig, estimate_road_geometry, median_background
    from dataclasses import asdict
    out_dir = os.path.join(CACHE_DIR, "collision_suite", split)
    sc, meta = sample_scenario(split, i)
    vp, gp, gt = render_scenario(sc, out_dir) if not os.path.exists(os.path.join(out_dir, sc.name + "_ground_truth.json")) else \
        (os.path.join(out_dir, sc.name + ".mp4"), None, json.load(open(os.path.join(out_dir, sc.name + "_ground_truth.json"))))
    fps, frames, info = ingest(vp)
    prior = RoadPrior()
    rng = np.random.default_rng(i)
    geo = estimate_road_geometry(frames, prior, GeometryConfig(seed=i), background=median_background(frames))
    case = {"pipeline_version": PIPELINE_VERSION, "input": {"file": os.path.basename(vp), **info}, "prior": asdict(prior),
            "stages": [], "oracle_tracks": "occluded" if occluded else "clean"}
    if not geo.ok:
        return {"status": "NOT_RECONSTRUCTABLE", "verdict": None}, gt, sc, meta
    case["geometry"] = geo.to_json()
    tracks = oracle_tracks(gt, sc, occluded, rng)
    out, _ = analyze_from_tracks(case, frames, fps, geo, tracks, prior, rng, 80, None, {}, time.time())
    return out, gt, sc, meta


def gt_near_miss(sc, t0, t1, min_closing=0.3, ttc_cap=10.0):
    """Ground-truth surrogate safety measures over [t0, t1] (CARAI's observed interval), from the
    simulator's vehicle states: bumper gap = d (x_lead - x_follow) - (L_lead + L_follow) / 2."""
    lead = next(v for v in sc.vehicles if v.vid == "lead")
    fol = next(v for v in sc.vehicles if v.vid == "follow")
    d = lead.direction
    tt = np.arange(t0, t1 + 1e-9, 0.01)
    xl, vl, _ = _free(lead, tt)
    xf, vf, _ = _free(fol, tt)
    gap = d * (xl - xf) - (lead.length_m + fol.length_m) / 2.0
    clo = vf - vl                     # simulate_vehicle returns speeds (magnitudes along the travel direction)
    m = (clo > min_closing) & (gap > 0)
    ttc = float((gap[m] / clo[m]).min()) if m.any() else np.inf
    return {"min_gap_m": float(gap.min()), "min_ttc_s": float(min(ttc, ttc_cap)),
            "max_drac_mps2": float(np.max(clo[m] ** 2 / (2 * gap[m]))) if m.any() else 0.0}


def score_near_miss(case, sc):
    nm = case.get("near_miss")
    if not nm or "observed_interval_s" not in nm:
        return None
    g = gt_near_miss(sc, *nm["observed_interval_s"], ttc_cap=nm.get("ttc_cap_s", 10.0))
    res = {"gt": g}
    for k in ("min_gap_m", "min_ttc_s", "max_drac_mps2"):
        if k in nm and k in g:
            lo, hi = nm[k]["ci90"]
            res[k] = {"est": nm[k]["median"], "gt": g[k], "err": nm[k]["median"] - g[k], "covered": bool(lo <= g[k] <= hi)}
    return res


def score_case(row, case, gt, sc, meta):
    row["status"] = case.get("status")
    v = (case.get("verdict") or {}).get("value")
    if case.get("status") in ("NOT_RECONSTRUCTABLE", "INVALID_INPUT", "MOVING_CAMERA"):
        v = "NOT_RECONSTRUCTABLE"
    row["carai_verdict"] = v
    row["skeptic_disputed"] = (case.get("skeptic") or {}).get("disputed_checks")
    row["stages"] = case.get("stages")
    col = case.get("collision") or {}
    row["carai_contact_t_s"] = col.get("t_contact_s")
    row["carai_contact_ci90"] = col.get("t_contact_ci90_s")
    row["contact_source"] = col.get("source")
    hz = case.get("hazard_onset") or {}
    row["carai_hazard"] = {k: hz.get(k) for k in ("source", "t_s", "t_lo_s", "t_hi_s")}
    cf = case.get("counterfactual")
    if cf:
        row["carai_cf_verdict"] = cf["verdict"]
        row["carai_crit"] = cf["critical_reaction_s"]
        row["carai_p_avoid"] = cf["p_avoid"]
        p = cf["p_avoid"][DECEL_GRID.index(7.0)][REACTION_GRID.index(1.25)]
        row["point_assumption_verdict"] = "AVOIDABLE" if p >= 0.5 else "NOT_AVOIDABLE"
    row["follower_speed_at_onset"] = case.get("follower_speed_at_hazard_onset")
    row["gt_observable"] = gt_observable(gt, sc, row.get("gt_collision_t_s"))
    try:
        row["event_resolution"] = event_resolution(gt, sc, row.get("gt_collision_t_s"))
    except Exception:  # noqa: BLE001 -- stratification only
        row["event_resolution"] = None
    try:
        row["near_miss"] = score_near_miss(case, sc)
    except Exception as e:  # noqa: BLE001 -- secondary metric
        row["near_miss"] = {"error": str(e)[:200]}
    row["outcome"] = outcome(row)
    return row


def run_one(args):
    split, i = args[:2]
    mode = args[2] if len(args) > 2 else "e2e"
    if mode in ("oracle_clean", "oracle_occluded"):
        row = {"split": split, "i": i, "mode": mode}
        try:
            case, gt, sc, meta = run_oracle(split, i, mode == "oracle_occluded")
            row.update({"name": sc.name, **meta})
            gtc = gt.get("collision")
            row["gt_collision_t_s"] = gtc["t_s"] if gtc else None
            row["gt"] = gt_counterfactual(sc, meta, gtc["t_s"] if gtc else None)
            score_case(row, case, gt, sc, meta)
        except Exception:  # noqa: BLE001
            row["error"] = traceback.format_exc()[-1500:]
        return row
    from backend.pipeline.analyze import analyze_video
    out_dir = os.path.join(CACHE_DIR, "collision_suite", split)
    sc, meta = sample_scenario(split, i)
    row = {"split": split, "i": i, "name": sc.name, "mode": "e2e", **meta}
    try:
        t0 = time.time()
        vp, gp, gt = render_scenario(sc, out_dir)
        row["render_s"] = round(time.time() - t0, 1)
        gtc = gt.get("collision")
        row["gt_collision_t_s"] = gtc["t_s"] if gtc else None
        row["gt"] = gt_counterfactual(sc, meta, gtc["t_s"] if gtc else None)
        t0 = time.time()
        case = analyze_video(vp, seed=i)
        row["pipeline_s"] = round(time.time() - t0, 1)
        score_case(row, case, gt, sc, meta)
        g = case.get("geometry") or {}
        row["geometry_ok"] = bool(g.get("ok"))
        row["roll_est_deg"] = g.get("roll_deg")
        row["true_roll_deg"] = sc.camera.roll_deg
        # legacy pipeline on the same clip (ablation)
        try:
            if os.environ.get("CARAI_SKIP_LEGACY"):
                raise RuntimeError("skipped (--no-legacy)")
            from backend.api.process_video import process_video as legacy
            t0 = time.time()
            lr, _ = legacy(vp)
            row["legacy_s"] = round(time.time() - t0, 1)
            row["legacy_status"] = lr.get("status")
            row["legacy_verdict"] = (lr.get("causality") or {}).get("verdict")
        except Exception as e:  # noqa: BLE001
            row["legacy_error"] = str(e)[:200]
    except Exception as e:  # noqa: BLE001
        row["error"] = traceback.format_exc()[-1500:]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mode", default="e2e", choices=["e2e", "oracle_clean", "oracle_occluded"])
    ap.add_argument("--out", default=None, help="results file (default: evaluation/results/collision_suite_<split>[_<mode>].jsonl)")
    ap.add_argument("--no-legacy", action="store_true", help="skip the v1-pipeline ablation (faster)")
    args = ap.parse_args()
    if args.no_legacy:
        os.environ["CARAI_SKIP_LEGACY"] = "1"
    suffix = "" if args.mode == "e2e" else f"_{args.mode}"
    out = args.out or os.path.join(RESULTS_DIR, f"collision_suite_{args.split}{suffix}.jsonl")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["i"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [(args.split, i, args.mode) for i in range(args.start, args.start + args.n) if i not in done]
    with Pool(args.workers) as pool, open(out, "a") as f:
        for row in pool.imap_unordered(run_one, todo):
            f.write(json.dumps(row, default=float) + "\n")
            f.flush()
            print(f"{row.get('name')}: gt={row.get('gt', {}).get('verdict')} carai={row.get('carai_verdict')} "
                  f"outcome={row.get('outcome')} legacy={row.get('legacy_verdict')} "
                  f"t={row.get('pipeline_s')}s err={'Y' if 'error' in row else ''}", flush=True)


if __name__ == "__main__":
    main()
