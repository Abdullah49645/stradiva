"""
The investigator: what Stradiva does when its own Skeptic disputes a
reconstruction, instead of stopping there.

Loop: PERCEIVE (read the case and the Skeptic's verdicts) -> DIAGNOSE (map
a blocking dispute to the failure it names) -> ACT (run the one remedy that
failure admits, with the production pipeline) -> RE-EXAMINE (the Skeptic
cross-examines the new reconstruction from scratch) -> ADOPT or KEEP the
original refusal. Every step is recorded in case["investigation"]["trace"].

Remedies, one per failure the Skeptic can name:

  temporal_stability  "the camera moved during the clip". Find the abrupt
                      change in the static background (feature tracking +
                      a similarity transform against the first frame),
                      split the clip there, and analyse each stable part
                      on its own. Refuses if the motion is gradual (zoom,
                      drift) or if no part is long enough.
  marking_standard    "the dashes do not have the assumed proportions".
                      Look up every documented standard whose duty cycle
                      matches the measured one (backend/geometry/
                      marking_standards.py), re-run the geometry and the
                      reasoning under each, and issue a verdict only if it
                      is the same under all of them. Speeds are reported
                      as the union of their intervals: the cycle length is
                      still not measurable, so the spread is real.
  kinematics          "a subject vehicle's reconstructed motion is
                      physically impossible". This is a tracking failure
                      at specific frames (a blob that fused with a
                      neighbour, a ghost). Those measurements are
                      quarantined (treated as unmeasured, as under
                      occlusion) and the pipeline re-runs on the rest.
                      Adopted only if the Skeptic then passes, at most
                      MAX_QUARANTINE_FRAC of a vehicle's measurements were
                      removed, and none of them lies within
                      CONTACT_GUARD_S of the contact the re-run reports.
  held_out_markings   no automatic remedy (the marking evidence itself is
                      inconsistent): the refusal stands, with a request.

What the investigator never does: change a threshold, drop a Skeptic check,
or pick among candidate answers by which one looks better. Inputs it cannot
infer (frame rate, marking standard, focal length) are listed as requests to
the user, with the reason, rather than guessed.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict

import cv2
import numpy as np

from backend.geometry.marking_standards import BY_KEY, DUTY_INFLATION_MAX, candidates_for_duty, consistent_with_duty, prior_for
from backend.geometry.road_geometry import GeometryConfig, RoadPrior, estimate_road_geometry, median_background
from backend.pipeline.analyze import BLOCKING_CHECKS, analyze_from_tracks, analyze_video

MAX_QUARANTINE_FRAC = 0.25
NEAR_GAP_M = 3.0               # impossible motion while the pair is this close may be the impact itself
REMAINDER_ACCEL_MAX = 6.0      # after quarantine the subject's motion must be clearly physical (half the 12 m/s^2 limit)
CONTACT_GUARD_S = 0.3
QUARANTINE_ROUNDS = 3
SHIFT_PX = 2.0                 # background displacement that counts as "the camera moved"
MIN_SEGMENT_S = 1.5
DUTY_TOL = 0.06                # the Skeptic's own marking_standard tolerance
ORDER = ["temporal_stability", "marking_standard", "kinematics", "held_out_markings"]
INFORMATIVE = {"RECONSTRUCTED_COLLISION": 4, "RECONSTRUCTED_NO_COLLISION": 3, "CONTACT_UNDETERMINED": 2,
               "RECONSTRUCTED_NO_INTERACTION": 1}


def _summary(case):
    v = (case.get("verdict") or {}).get("value")
    sk = case.get("skeptic") or {}
    return {"status": case.get("status"), "verdict": v, "disputed_checks": sk.get("disputed_checks"),
            "contact_t_s": (case.get("collision") or {}).get("t_contact_s")}


PAIR_STATUSES = {"RECONSTRUCTED_COLLISION", "RECONSTRUCTED_NO_COLLISION", "CONTACT_UNDETERMINED"}


def _adoptable(case):
    """A remedy's result replaces a refusal only if every blocking check holds AND a same-lane pair was
    reconstructed: with no pair, the kinematics check can only test the majority of trajectories, which is
    too weak to stand in for the check that blocked the original result."""
    return not _blocking(case) and case.get("status") in PAIR_STATUSES


def _blocking(case):
    sk = case.get("skeptic") or {}
    b = [c for c in (sk.get("disputed_checks") or []) if c in BLOCKING_CHECKS]
    return sorted(b, key=lambda c: ORDER.index(c) if c in ORDER else 99)


def _settle(path, out, inn, ctx):
    """A re-run whose ONLY blocking dispute is kinematics gets the impossible-motion remedy too
    (a wrong scale or a camera bump also produces impossible accelerations, so the two often come
    together). Returns (case, internals, sub_step or None)."""
    if inn is None or _blocking(out) != ["kinematics"]:
        return out, inn, None
    res, sub = remedy_impossible_motion(path, out, inn, ctx)
    if res is None:
        return out, inn, sub
    return res[0], res[1], sub


# ------------------------------------------------------------ remedy: camera moved --

def find_camera_shift(frames, fps, step_s=0.1):
    """Displacement (px, worst image corner) of the scene in each sampled frame relative to the
    median of the first 0.3 s, by ECC image alignment (Euclidean motion: rotation + shift) of
    blurred, downscaled grey images. ECC uses every pixel (markings, road edges, texture), so it
    works on low-texture scenes where corner tracking does not; moving vehicles cover a small part
    of the image. Returns (change_frame or None, series [(frame, px)], reason)."""
    h, w = frames[0].shape[:2]
    s = min(1.0, 480.0 / w)

    def g(f):
        return cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                                             cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    n0 = max(3, int(round(0.3 * fps)))
    ref = np.median(np.stack([g(f) for f in frames[:n0]]), axis=0).astype(np.float32)
    corners = np.array([[0, 0], [w * s, 0], [0, h * s], [w * s, h * s]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)

    def disp(k):
        M = np.eye(2, 3, dtype=np.float32)
        try:
            _, M = cv2.findTransformECC(ref, g(frames[k]), M, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        except cv2.error:
            return None
        moved = corners @ M[:, :2].T + M[:, 2] - corners
        return float(np.max(np.linalg.norm(moved, axis=1)) / s)

    step = max(1, int(round(step_s * fps)))
    series = [(k, disp(k)) for k in range(0, len(frames), step)]
    valid = [(k, d) for k, d in series if d is not None]
    if len(valid) < 5:
        return None, series, "the scene could not be aligned through the clip"
    ds = np.array([d for _, d in valid])
    kk = np.array([k for k, _ in valid])
    above = ds > SHIFT_PX
    if not above.any():
        return None, series, f"the scene never moved by more than {SHIFT_PX} px: no camera bump"
    first = int(np.argmax(above))
    if first == 0 or above[first:].mean() < 0.9 or above[:first].any():
        return None, series, "the scene moves back and forth: not a single camera bump"
    after = ds[first:]
    if np.ptp(np.percentile(after, [10, 90])) > SHIFT_PX:
        return None, series, "the scene keeps moving after the first shift (zoom or drift, not a single bump)"
    lo, hi = int(kk[first - 1]), int(kk[first])
    for k in range(lo + 1, hi + 1):
        d = disp(k)
        if d is not None and d > SHIFT_PX:
            hi = k
            break
    return hi, series, (f"the scene shifted by {np.median(after):.1f} px at frame {hi} ({hi / fps:.2f} s) "
                        "and stayed there")


def _shift_times(case, dt):
    if not dt:
        return case
    col = case.get("collision") or {}
    for k in ("t_contact_s", "t_contact_extrapolated_s", "lead_last_observed_s"):
        if col.get(k) is not None:
            col[k] += dt
    if col.get("t_contact_ci90_s"):
        col["t_contact_ci90_s"] = [x + dt for x in col["t_contact_ci90_s"]]
    hz = case.get("hazard_onset") or {}
    for k in ("t_s", "t_lo_s", "t_hi_s"):
        if hz.get(k) is not None:
            hz[k] += dt
    for sp in case.get("speeds") or []:
        sp["t_ref_s"] = sp["t_ref_s"] + dt
    for v in case.get("vehicles") or []:
        v["t_s"] = [t + dt for t in v["t_s"]]
    nm = case.get("near_miss") or {}
    if nm.get("observed_interval_s"):
        nm["observed_interval_s"] = [t + dt for t in nm["observed_interval_s"]]
    if nm.get("t_min_ttc_s") is not None:
        nm["t_min_ttc_s"] += dt
    case["time_base"] = f"times shifted by +{dt:.3f} s to the start of the original clip"
    return case


def remedy_camera_moved(path, case, internals, ctx):
    frames, fps = internals["frames"], internals["fps"]
    cp, series, why = find_camera_shift(frames, fps)
    step = {"diagnosis": "temporal_stability: the geometry of the two halves of the clip disagrees, so a single "
                         "road geometry cannot be right for the whole clip",
            "action": "measure the background's displacement through the clip and look for one abrupt shift",
            "observation": why}
    if cp is None:
        step["decision"] = "no single camera bump found: the dispute stands"
        return None, step
    guard = max(2, int(round(0.1 * fps)))
    segs = [(0, cp), (cp + guard, len(frames))]
    min_n = max(40, int(MIN_SEGMENT_S * fps))
    results = []
    for a, b in segs:
        if b - a < min_n:
            results.append({"frames": [a, b], "skipped": f"shorter than {MIN_SEGMENT_S} s"})
            continue
        r, ri = analyze_video(path, ctx["prior"], ctx["n_mc"], ctx["seed"], fps_override=ctx.get("fps"),
                              frame_range=(a, b), return_internals=True, preloaded=internals.get("ingest"),
                              **ctx.get("kw", {}))
        r, ri, sub = _settle(path, r, ri, ctx)
        results.append({"frames": [a, b], "t_s": [a / fps, b / fps], **_summary(r), "_case": r, "_int": ri,
                        **({"sub_step": sub} if sub else {})})
    usable = [r for r in results if "_case" in r and _adoptable(r["_case"])]
    step["action"] += f"; split at frame {cp} and analyse each part separately"
    step["segments"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    if not usable:
        step["decision"] = ("no part of the clip gives an undisputed reconstruction of the two vehicles: "
                            "the dispute stands")
        return None, step
    best = max(usable, key=lambda r: (INFORMATIVE.get(r["status"], 0), r["frames"][1] - r["frames"][0]))
    out = _shift_times(best["_case"], best["frames"][0] / fps)
    out["segment_analysis"] = {"camera_shift_frame": cp, "camera_shift_t_s": cp / fps,
                               "segments": step["segments"], "adopted_frames": best["frames"]}
    step["decision"] = (f"adopted the reconstruction of frames {best['frames'][0]}-{best['frames'][1]} "
                        f"({best['status']}); the other part is reported separately")
    return (out, None), step      # no further remedies on a segment (its own Skeptic run already passed)


# ------------------------------------------------------- remedy: marking standard --

def _rerun_with_prior(internals, prior_c):
    frames, fps = internals["frames"], internals["fps"]
    seed = internals["seed"]
    geo = estimate_road_geometry(frames, prior_c, GeometryConfig(seed=seed), background=median_background(frames))
    case = copy.deepcopy(internals["case_pre"])
    case["prior"] = asdict(prior_c)
    case["geometry"] = geo.to_json()
    if not geo.ok:
        return {**case, "status": "NOT_RECONSTRUCTABLE", "reasons": geo.reasons}, None
    tracks = copy.deepcopy(internals["tracks"])
    out, inn = analyze_from_tracks(case, frames, fps, geo, tracks, prior_c, np.random.default_rng(seed),
                                   internals["n_mc"], internals["skeptic_checks"], {}, time.time())
    inn.update({k: internals[k] for k in ("frames", "case_pre", "seed", "n_mc", "skeptic_checks")})
    inn["prior"] = prior_c
    inn["case_pre"] = {**copy.deepcopy(internals["case_pre"]), "prior": asdict(prior_c), "geometry": geo.to_json()}
    return out, inn


def remedy_marking_standard(path, case, internals, ctx):
    chk = next(c for c in case["skeptic"]["checks"] if c["check"] == "marking_standard")
    dm = chk["statistic"]["duty_measured"]
    cands = candidates_for_duty(dm, DUTY_TOL, exclude=internals["prior"].name)
    step = {"diagnosis": f"marking_standard: dashes cover {dm:.0%} of each cycle; the assumed standard "
                         f"({internals['prior'].name}) says {chk['statistic']['duty_prior']:.0%}",
            "action": (f"look up documented standards whose duty cycle is consistent with the measured one (within "
                       f"{DUTY_TOL:.2f}, allowing up to {DUTY_INFLATION_MAX:.2f} of upward measurement bias)"),
            "observation": (", ".join(f"{c.name} ({c.duty:.0%})" for c in cands) or "none in the library")}
    other = [c for c in _blocking(case) if c in ("temporal_stability", "held_out_markings")]
    if other:
        step["decision"] = ("the marking evidence itself is in question (" + ", ".join(other) + "), so its measured "
                            "proportions cannot be used to choose another standard; the dispute stands")
        return None, step
    pd = internals["prior"].dash_len_m / internals["prior"].cycle_m
    if consistent_with_duty(pd, dm, DUTY_TOL):
        step["decision"] = (f"allowing for the measurement's known upward bias (up to {DUTY_INFLATION_MAX:.2f}), the assumed "
                            "standard cannot be ruled out, and under it the reconstruction is disputed; the dispute stands")
        ctx["requests"].append({"input": "marking_standard", "why": "the dash proportions are borderline for the assumed "
                                "standard; the result depends on which standard applies here",
                                "options": [internals["prior"].name] + [c.key for c in cands]})
        return None, step
    if not cands:
        step["decision"] = "no documented standard matches: the dispute stands; the user is asked for the standard"
        ctx["requests"].append({"input": "marking_standard", "why": f"the road's dashes cover {dm:.0%} of each cycle, "
                                "which matches no standard Stradiva knows", "options": sorted(BY_KEY)})
        return None, step
    runs = []
    for c in cands:
        out, inn = _rerun_with_prior(internals, prior_for(c, internals["prior"]))
        out, inn, sub = _settle(path, out, inn, ctx)
        runs.append({"standard": c.key, "name": c.name, "cycle_m": c.cycle_m, "source": c.source,
                     **_summary(out), "_case": out, "_int": inn, **({"sub_step": sub} if sub else {})})
    step["action"] += "; re-run the geometry and the reasoning under each"
    step["candidates"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]
    ok = [r for r in runs if r["_int"] is not None and _adoptable(r["_case"])]
    if len(ok) < len(runs):
        bad = [r["name"] for r in runs if r not in ok]
        step["decision"] = f"the reconstruction is not undisputed under every matching standard ({', '.join(bad)}): the dispute stands"
        ctx["requests"].append({"input": "marking_standard", "why": "the answer depends on which matching standard "
                                "applies at this location", "options": [r["standard"] for r in runs]})
        return None, step
    verdicts = {r["verdict"] for r in ok}
    adopted = ok[0]
    out = adopted["_case"]
    # speeds: union of the intervals under every matching standard (the cycle length is not measurable)
    by_id = {}
    for r in ok:
        for sp in r["_case"].get("speeds") or []:
            by_id.setdefault(sp["vehicle_id"], []).append(sp)
    for sp in out.get("speeds") or []:
        alts = by_id.get(sp["vehicle_id"], [])
        cis = [a["ci90_mps"] for a in alts if a.get("ci90_mps")]
        if cis:
            sp["ci90_mps_single_standard"] = sp.get("ci90_mps")
            sp["ci90_mps"] = [min(c[0] for c in cis), max(c[1] for c in cis)]
            sp["speed_mps_by_standard"] = {r["standard"]: next((a["speed_mps"] for a in (r["_case"].get("speeds") or [])
                                                                if a["vehicle_id"] == sp["vehicle_id"]), None) for r in ok}
    # every other interval the UI shows is widened the same way
    fs = [r["_case"].get("follower_speed_at_hazard_onset") for r in ok]
    if all(f and f.get("ci90_mps") for f in fs) and out.get("follower_speed_at_hazard_onset"):
        out["follower_speed_at_hazard_onset"]["ci90_mps"] = [min(f["ci90_mps"][0] for f in fs), max(f["ci90_mps"][1] for f in fs)]
    cfs = [r["_case"].get("counterfactual") for r in ok]
    if all(cfs) and out.get("counterfactual"):
        for a, cr in out["counterfactual"]["critical_reaction_s"].items():
            alts = [c["critical_reaction_s"].get(a) for c in cfs if c["critical_reaction_s"].get(a)]
            cr["ci90_s"] = [min(x["ci90_s"][0] for x in alts), max(x["ci90_s"][1] for x in alts)]
    nms = [r["_case"].get("near_miss") for r in ok]
    if out.get("near_miss") and all(n and "min_gap_m" in n for n in nms):
        for k in ("min_gap_m", "min_ttc_s", "max_drac_mps2"):
            if all(k in n for n in nms):
                out["near_miss"][k]["ci90"] = [min(n[k]["ci90"][0] for n in nms), max(n[k]["ci90"][1] for n in nms)]
    out["interval_note"] = ("intervals span every documented marking standard whose proportions match the road: "
                            + ", ".join(r["name"] for r in ok))
    out["marking_standard_candidates"] = step["candidates"]
    if len(verdicts) > 1:
        out["verdict"] = {"value": "INSUFFICIENT_EVIDENCE",
                          "reason": "the verdict depends on which of the matching marking standards applies: "
                                    + "; ".join(f"{r['name']}: {r['verdict']}" for r in ok)}
        out["status"] = "STANDARD_DEPENDENT"
        ctx["requests"].append({"input": "marking_standard", "why": "the verdict differs between the standards that "
                                "match the road's dash proportions", "options": [r["standard"] for r in ok]})
        step["decision"] = "the standards disagree on the verdict: no verdict; speeds reported with the union interval"
    else:
        step["decision"] = (f"the verdict ({adopted['verdict']}) is the same under all {len(ok)} matching standard(s); "
                            "adopted, with speed intervals spanning all of them")
        if len(ok) == 1:
            ctx["requests"].append({"input": "marking_standard", "why": f"one documented standard ({adopted['name']}) "
                                    "matches the dash proportions; the user should confirm it applies here",
                                    "options": [adopted["standard"]]})
    return (out, adopted["_int"]), step


# ------------------------------------------------------- remedy: impossible motion --

def _implausible_frames(kin, fps, t_cut, win=0.4, a_max=12.0, v_max=70.0):
    t, v, sv = np.asarray(kin["t"]), np.asarray(kin["v"]), np.asarray(kin["sv"])
    m = t < (t_cut - 0.3) if t_cut else np.ones(len(t), bool)
    t, v, sv = t[m], v[m], sv[m]
    bad = set()
    if len(t) < 3:
        return bad
    j = np.searchsorted(t, t + win)
    ok = j < len(t)
    i, j = np.where(ok)[0], j[ok]
    dt = t[j] - t[i]
    a = (v[j] - v[i]) / dt
    sa = np.sqrt(sv[i] ** 2 + sv[j] ** 2) / dt
    for ii, jj in zip(i[np.abs(a) - 2 * sa > a_max], j[np.abs(a) - 2 * sa > a_max]):
        bad |= set(range(int(round(t[ii] * fps)), int(round(t[jj] * fps)) + 1))
    for ii in np.where(np.abs(v) - 2 * sv > v_max)[0]:
        bad.add(int(round(t[ii] * fps)))
    return bad


def _pair_kins(inn):
    pair = inn.get("pair")
    if not pair or len(pair) < 2:
        return None
    return pair[0]["kin"], pair[1]["kin"]


def _near_event(frames_bad, fps, pair_k):
    """Reason string if any quarantine candidate frame is within NEAR_GAP_M of contact or in the last 0.5 s of the
    pair's common observation (impacts and occlusions happen there), else None."""
    if not frames_bad or pair_k is None:
        return None
    kf, kl = pair_k
    t_bad = np.array(sorted(frames_bad), float) / fps
    t0, t1 = max(kf["t"][0], kl["t"][0]), min(kf["t"][-1], kl["t"][-1])
    if t1 <= t0:
        return None
    tb = t_bad[(t_bad >= t0) & (t_bad <= t1)]
    if not len(tb):
        return None
    gap = np.interp(tb, kl["t"], kl["x"]) - np.interp(tb, kf["t"], kf["x"]) - 4.6
    if gap.min() < NEAR_GAP_M:
        return f"at t = {tb[np.argmin(gap)]:.2f} s the reconstructed gap is {gap.min():.1f} m"
    if tb.max() > t1 - 0.5:
        return f"at t = {tb.max():.2f} s, within 0.5 s of the end of the pair's common observation ({t1:.2f} s)"
    return None


def remedy_impossible_motion(path, case, internals, ctx):
    fps = internals["fps"]
    chk = next(c for c in case["skeptic"]["checks"] if c["check"] == "kinematics")
    col = case.get("collision") or {}
    subjects = {col.get("follower_id"), col.get("lead_id")} - {None}
    step = {"diagnosis": "kinematics: " + chk["evidence"],
            "action": "quarantine the measurements of the subject vehicle(s) in the frames where the motion is "
                      "impossible (treat them as unmeasured) and re-run the reasoning on the rest"}
    tracks = copy.deepcopy(internals["tracks"])
    quarantined = {}
    out, inn = case, internals
    for rnd in range(QUARANTINE_ROUNDS):
        kchk = next((c for c in out["skeptic"]["checks"] if c["check"] == "kinematics"), None)
        if kchk is None or kchk["status"] != "DISPUTE":
            break
        t_cut = (out.get("collision") or {}).get("t_contact_s")
        bad_ids = {p["vehicle_id"] for p in kchk["statistic"]["vehicles"] if not p["plausible"]}
        subj = {(out.get("collision") or {}).get("follower_id"), (out.get("collision") or {}).get("lead_id")}
        new = 0
        pair_k = _pair_kins(inn)
        for veh in inn["vehicles"]:
            vid = veh["track"]["id"]
            if vid not in bad_ids or vid not in subj:
                continue
            frames_bad = _implausible_frames(veh["kin"], fps, t_cut)
            near = _near_event(frames_bad, fps, pair_k)
            if near:
                step["observation"] = near
                step["decision"] = ("the impossible motion happens while the two vehicles are close or at the end of their "
                                    "common observation: it may be the collision itself, so it is not removed; the dispute stands")
                return None, step
            ids = set(veh["track"].get("stitched_from", [vid]))
            for tr in tracks:
                if tr["id"] not in ids:
                    continue
                for r in tr["records"]:
                    if r["frame"] in frames_bad and r["status"] in ("MEASURED", "MEASURED_OCCLUDING"):
                        r["status"] = "QUARANTINED"
                        quarantined.setdefault(vid, set()).add(r["frame"])
                        new += 1
        if not new:
            break
        c0 = copy.deepcopy(internals["case_pre"])
        out, inn2 = analyze_from_tracks(c0, internals["frames"], fps, inn["geo"], copy.deepcopy(tracks), internals["prior"],
                                        np.random.default_rng(internals["seed"]), internals["n_mc"],
                                        internals["skeptic_checks"], {}, time.time())
        inn2.update({k: internals[k] for k in ("frames", "case_pre", "seed", "n_mc", "skeptic_checks", "prior")})
        inn = inn2
    n_meas = {tr["id"]: sum(r["status"] in ("MEASURED", "MEASURED_OCCLUDING") for r in tr["records"])
              for tr in internals["tracks"]}
    groups = {v["track"]["id"]: v["track"].get("stitched_from", [v["track"]["id"]]) for v in internals["vehicles"]}
    frac = {vid: len(fr) / max(1, sum(n_meas.get(i, 0) for i in groups.get(vid, [vid])))
            for vid, fr in quarantined.items()}
    step["observation"] = ("; ".join(f"vehicle {vid}: {len(fr)} measurement(s) quarantined ({frac[vid]:.0%}), "
                                     f"t = {min(fr) / fps:.2f}-{max(fr) / fps:.2f} s" for vid, fr in quarantined.items())
                           or "no measurement could be attributed to the impossible motion")
    step["re_examination"] = _summary(out)
    if not quarantined:
        step["decision"] = "nothing to quarantine: the dispute stands"
        return None, step
    if _blocking(out):
        step["decision"] = "the Skeptic still disputes the re-run (" + ", ".join(_blocking(out)) + "): the dispute stands"
        return None, step
    if not _adoptable(out):
        step["decision"] = ("without the quarantined measurements no same-lane pair is reconstructed, so the checks "
                            "that blocked the result cannot be re-applied: the dispute stands")
        return None, step
    if max(frac.values()) > MAX_QUARANTINE_FRAC:
        step["decision"] = (f"more than {MAX_QUARANTINE_FRAC:.0%} of a vehicle's measurements would have to be discarded: "
                            "too little of the track is trustworthy; the dispute stands")
        return None, step
    kchk = next((c for c in out["skeptic"]["checks"] if c["check"] == "kinematics"), None)
    worst = max((p["max_abs_accel_mps2"] for p in (kchk or {}).get("statistic", {}).get("vehicles", [])
                 if p["vehicle_id"] in quarantined), default=0.0)
    if worst > REMAINDER_ACCEL_MAX:
        step["decision"] = (f"after quarantine a subject vehicle still needs {worst:.1f} m/s^2 (lower bound): passes the "
                            f"{12:g} m/s^2 limit, but not clearly enough (<= {REMAINDER_ACCEL_MAX:g}) to trust the rest of a "
                            "track already shown to be mistracked; the dispute stands")
        return None, step
    tc = (out.get("collision") or {}).get("t_contact_s")
    if tc is not None and any(abs(f / fps - tc) < CONTACT_GUARD_S for fr in quarantined.values() for f in fr):
        step["decision"] = ("quarantined measurements lie within "
                            f"{CONTACT_GUARD_S} s of the contact the re-run reports: the contact would rest on "
                            "the measurements in question; the dispute stands")
        return None, step
    out["quarantine"] = {str(vid): {"frames": sorted(fr), "fraction": frac[vid]} for vid, fr in quarantined.items()}
    step["decision"] = "adopted: the Skeptic passes the re-run, which does not rest on the quarantined measurements"
    return (out, inn), step


REMEDIES = {"temporal_stability": remedy_camera_moved, "marking_standard": remedy_marking_standard,
            "kinematics": remedy_impossible_motion}


# --------------------------------------------------------------------- driver --

def investigate(path, prior: RoadPrior = None, seed=0, n_mc=80, user_inputs=None, max_steps=3, return_base=False,
                max_seconds=None):
    """Analyse a clip and, if the Skeptic blocks the result, try the remedy for the failure it
    names. Returns the adopted case, with case["investigation"] holding the full trace."""
    user_inputs = user_inputs or {}
    t0 = time.time()
    prior = prior or RoadPrior()
    trace, requests = [], []
    if user_inputs.get("marking_standard"):
        prior = prior_for(BY_KEY[user_inputs["marking_standard"]], prior)
        trace.append({"step": "input", "observation": f"marking standard supplied by the user: {prior.name}"})
    fps = user_inputs.get("fps")
    kw = {"max_seconds": max_seconds} if max_seconds else {}
    base, internals = analyze_video(path, prior, n_mc, seed, return_internals=True, fps_override=fps,
                                    focal_px=user_inputs.get("focal_px"), **kw)
    trace.append({"step": "analyse", "action": "full pipeline with the Skeptic", "observation": _summary(base)})
    ctx = {"prior": prior, "n_mc": n_mc, "seed": seed, "fps": fps, "requests": requests, "kw": kw}
    case, inn = base, internals
    tried = set()
    for _ in range(max_steps):
        if inn is None:
            break
        todo = [c for c in _blocking(case) if c not in tried]
        if not todo:
            break
        chk = todo[0]
        tried.add(chk)
        rem = REMEDIES.get(chk)
        if rem is None:
            trace.append({"step": "diagnose", "diagnosis": chk, "decision": "no automatic remedy for this failure: "
                          "the refusal stands"})
            if chk == "held_out_markings":
                requests.append({"input": "footage", "why": "the road markings are not consistent with any single "
                                 "regular pattern (worn, missing or irregular dashes in view)"})
            continue
        res, step = rem(path, case, inn, ctx)
        step["step"] = f"remedy:{chk}"
        trace.append(step)
        if res is None:
            continue          # the next blocking dispute may still have a remedy; adoption always needs ALL to pass
        case, inn = res
    if base.get("input", {}).get("fps_source", "").startswith("container") and base.get("status") != "INVALID_INPUT":
        requests.append({"input": "fps", "why": f"every speed scales with the frame rate; {base['input'].get('fps')} fps "
                         "comes from the file's metadata, which is often wrong. Confirm it if you know the camera's rate"})
    if case.get("status") == "MOVING_CAMERA":
        dc = case.get("dashcam") or {}
        reasons = [r for w in dc.get("windows", []) for r in w.get("reasons", [])]
        if any("inconsistent with lane geometry" in r for r in reasons):
            requests.append({"input": "marking_standard", "why": "the assumed dash cycle contradicts the lane width "
                             "seen by the camera in some windows", "options": sorted(BY_KEY)})
        if not user_inputs.get("focal_px"):
            requests.append({"input": "focal_px", "why": "with the camera's focal length Stradiva can test the marking "
                             "standard against the lane width and measure the lead vehicle's range and closing speed"})
    case["investigation"] = {"base": _summary(base), "final": _summary(case), "trace": trace, "requests": requests,
                             "changed": case is not base, "duration_s": round(time.time() - t0, 2),
                             "user_inputs": user_inputs}
    return (case, base) if return_base else case
