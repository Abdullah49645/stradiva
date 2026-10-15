"""
CARAI production pipeline (single entry point used by the evaluation
harness, the demo-case generator, the local API and the AWS Lambda).

    analyze_video(path)  ->  case dict (JSON-safe)

No ground truth, no knowledge of the simulator, no scenario-specific
constants. Stages:

  0. ingest              decode, cap duration, downscale very large frames
  1. camera motion       static camera?  else -> dashcam ego-speed mode
  2. road geometry       backend/geometry/road_geometry.py (median background,
                         projective lines, closed-form roll, dash indexing,
                         bootstrap homographies)
  3. tracking            cv/tracking/mot.py (MOG2 + Kalman + Hungarian, merge-aware)
  4. kinematics          backend/reconstruction/kinematics.py (RTS smoother,
                         per-frame sigma from the homography Jacobian)
  5. events              same-lane pair, contact time (extrapolated from the
                         last reliable positions), lead brake onset (hinge fit)
  6. Monte Carlo         geometry bootstrap x marking prior x vehicle length x
                         onset uncertainty -> trajectory samples
  7. counterfactual      backend/causality/counterfactual.py (assumption envelope)
  8. cross-examination   backend/verification/skeptic.py (independent evidence;
                         blocking disputes withhold the verdict)
  9. identifiability     what the video measured, what was assumed, what it
                         cannot tell us
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict

import cv2
import numpy as np

from backend.geometry.road_geometry import (
    RoadPrior, GeometryConfig, estimate_road_geometry, image_to_road, road_to_image, median_background, event_background,
)
from cv.tracking.mot import TrackerConfig, track_vehicles, bottom_centre, contact_point, refine_background
from backend.reconstruction.kinematics import road_kinematics, velocity_kink
from backend.causality.counterfactual import TrajectorySample, assess as assess_counterfactual, hinge_onset
from backend.verification.skeptic import cross_examine

PIPELINE_VERSION = "carai-2.0"
BLOCKING_CHECKS = {"held_out_markings", "temporal_stability", "marking_standard", "kinematics"}
FOLLOW_LENGTH_PRIOR = (4.6, 0.35)     # passenger car length mean, sd (m) -- stated assumption
MAX_SECONDS = 20.0
MAX_WIDTH = 1280


# ------------------------------------------------------------------ ingest --

def ingest(path, max_seconds=MAX_SECONDS):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("could not open video (unsupported format or corrupt file)")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        fps_note = f"container reports fps={fps!r}; assuming 30"
        fps = 30.0
    else:
        fps_note = None
    frames, scale = [], 1.0
    max_n = int(max_seconds * fps)
    while len(frames) < max_n:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / f.shape[1]
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(f)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, frames, {"fps": fps, "fps_note": fps_note, "n_frames_used": len(frames),
                         "n_frames_container": int(total) if total and total > 0 else None,
                         "truncated": bool(total and total > len(frames)), "downscale": scale}


def is_static_camera(frames, n_pairs=6, gap_s=0.5, fps=30.0):
    """Fraction of corner features that do not move between frames gap_s apart."""
    n = len(frames)
    gap = max(1, int(gap_s * fps))
    if n <= gap:
        return True, 1.0
    fracs = []
    for i in np.linspace(0, n - 1 - gap, n_pairs).astype(int):
        g0 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(frames[i + gap], cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(g0, 300, 0.01, 8)
        if pts is None or len(pts) < 10:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None)
        ok = st.ravel() == 1
        if ok.sum() < 10:
            continue
        mv = np.linalg.norm((p1 - pts).reshape(-1, 2)[ok], axis=1)
        fracs.append(float(np.mean(mv < 1.0)))
    if not fracs:
        return True, None
    f = float(np.median(fracs))
    return f > 0.5, f


# ------------------------------------------------------------- utilities --

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _measured(track, fps):
    recs = [r for r in track["records"] if r["status"] in ("MEASURED", "MEASURED_OCCLUDING")]
    t = np.array([r["frame"] / fps for r in recs])
    uv = np.array([contact_point(r) for r in recs])
    boxes = [(r.get("contact_span"), r["contact_uv"][1]) for r in recs
             if r["status"] == "MEASURED" and r.get("contact_span")]   # occluding frames have no own footprint
    return t, uv, boxes


def _kin_for(track, fps, H, sx=1.0, sy=1.0, t_max=None):
    t, uv, _ = _measured(track, fps)
    if t_max is not None:
        keep = t <= t_max
        t, uv = t[keep], uv[keep]
    if len(t) < 6:
        return None
    k = road_kinematics(t, uv, H)
    if k is None:
        return None
    if sx != 1.0 or sy != 1.0:
        for key in ("x", "v", "a", "sx", "sv", "sa", "x_raw", "sigma_x_meas"):
            k[key] = k[key] * sx
        for key in ("y", "vy", "sy", "y_raw"):
            k[key] = k[key] * sy
    return k


def _lead_x(kin_l, a_lead, tq):
    """Lead rear-contact position: measured, then extrapolated at constant
    acceleration a_lead (<= 0) until rest."""
    tq = np.asarray(tq, float)
    out = np.interp(tq, kin_l["t"], kin_l["x"])
    after = tq > kin_l["t"][-1]
    if np.any(after):
        x0, v0 = kin_l["x"][-1], max(0.0, kin_l["v"][-1])
        dt = tq[after] - kin_l["t"][-1]
        if a_lead < -1e-6:
            ts = np.minimum(dt, v0 / -a_lead)
            out[after] = x0 + v0 * ts + 0.5 * a_lead * ts ** 2
        else:
            out[after] = x0 + v0 * dt
    return out


def _lead_v(kin_l, a_lead, t):
    if t <= kin_l["t"][-1]:
        return float(np.interp(t, kin_l["t"], kin_l["v"]))
    return max(0.0, float(kin_l["v"][-1]) + a_lead * (t - kin_l["t"][-1]))


def _extrapolate_contact(kin_f, kin_l, L_f, a_lead, horizon_s=0.2, dt=0.005):
    """First time the bumper gap x_lead - (x_follow + L_f) reaches 0. The
    follower is usually MEASURED up to contact (its ground contact stays
    visible while it occludes the lead); the lead is extrapolated from its
    last measured state with its measured deceleration. Returns (t, gap_at_
    last_lead_obs) or (None, ...)."""
    t_lo = max(kin_f["t"][0], kin_l["t"][0])
    t_hi = kin_f["t"][-1] + horizon_s
    tq = np.arange(t_lo, t_hi, dt)
    xf = np.interp(tq, kin_f["t"], kin_f["x"])
    after = tq > kin_f["t"][-1]
    if np.any(after):
        v0, a0 = max(0.0, kin_f["v"][-1]), min(0.0, kin_f["a"][-1])
        d = tq[after] - kin_f["t"][-1]
        ts = np.minimum(d, v0 / -a0) if a0 < -1e-6 else d
        xf[after] = kin_f["x"][-1] + v0 * ts + 0.5 * a0 * ts ** 2
    gap = _lead_x(kin_l, a_lead, tq) - (xf + L_f)
    hit = np.where(gap <= 0)[0]
    gap_last = float(np.interp(kin_l["t"][-1], tq, gap)) if kin_l["t"][-1] >= t_lo else None
    if len(hit) == 0:
        return None, gap_last
    return float(tq[hit[0]]), gap_last


def _lead_decel_from_impact(kin_f, kin_l, L_f, t_c):
    """While the lead is hidden (occluded) between its last observation t1 and
    the measured impact t_c, its rear must arrive exactly at the follower's
    front at t_c. Constant deceleration from (x1, v1) satisfying that, with
    the vehicle allowed to come to rest. Returns deceleration (>0) or None."""
    t1 = kin_l["t"][-1]
    x1, v1 = float(kin_l["x"][-1]), max(0.0, float(kin_l["v"][-1]))
    x_c = float(np.interp(t_c, kin_f["t"], kin_f["x"])) + L_f
    tau = t_c - t1
    d = x_c - x1
    if tau <= 0 or d <= 0:
        return None
    a = 2.0 * (d - v1 * tau) / tau ** 2          # constant-acceleration solution
    if a < 0 and v1 / -a < tau:                   # it would have stopped before t_c: stop at x_c instead
        a = -v1 ** 2 / (2.0 * d)
    return float(-a) if a < 0 else 0.0


def _json(o):
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_json(v) for v in o.tolist()]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# -------------------------------------------------------------------- main --

def analyze_video(path, prior: RoadPrior = None, n_mc=80, seed=0, skeptic_checks=None, max_seconds=MAX_SECONDS,
                  return_internals=False):
    t_start = time.time()
    prior = prior or RoadPrior()
    rng = np.random.default_rng(seed)
    timings = {}
    case = {"pipeline_version": PIPELINE_VERSION, "input": {"file": os.path.basename(path)},
            "prior": asdict(prior), "stages": []}
    try:
        case["input"]["sha256"] = _sha256(path)
    except OSError:
        pass

    def stage(name, status, detail=""):
        case["stages"].append({"stage": name, "status": status, "detail": detail,
                               "t_s": round(time.time() - t_start, 2)})

    # 0. ingest
    try:
        fps, frames, info = ingest(path, max_seconds)
    except ValueError as e:
        case.update({"status": "INVALID_INPUT", "reasons": [str(e)]})
        stage("ingest", "FAIL", str(e))
        return _json(case)
    case["input"].update(info)
    if len(frames) < 15:
        case.update({"status": "INVALID_INPUT", "reasons": [f"only {len(frames)} frames decoded (need >= 15)"]})
        stage("ingest", "FAIL", "too few frames")
        return _json(case)
    h, w = frames[0].shape[:2]
    case["input"].update({"width": w, "height": h})
    stage("ingest", "OK", f"{len(frames)} frames @ {fps:.2f} fps, {w}x{h}")
    timings["ingest"] = time.time() - t_start

    # 1. camera motion
    static, frac = is_static_camera(frames, fps=fps)
    case["camera_motion"] = {"static": static, "static_feature_fraction": frac}
    if not static:
        stage("camera_motion", "MOVING", "camera moves: routing to dashcam ego-speed analysis")
        case.update({"status": "MOVING_CAMERA", "mode": "dashcam",
                     "reasons": ["the camera moves (dashcam/handheld): third-party vehicle reconstruction needs a "
                                 "static camera; see the dashcam ego-speed result instead"]})
        try:
            from backend.dashcam.ego_speed import estimate_ego_speed, MarkingPrior
            mp = MarkingPrior("from case prior", prior.cycle_m, prior.cycle_rel_sigma)
            case["dashcam"] = estimate_ego_speed(path, mp, max_frames=len(frames))
        except Exception as e:  # noqa: BLE001
            case["dashcam_error"] = str(e)
        return _json(case)
    stage("camera_motion", "STATIC", f"{frac:.0%} of features static" if frac is not None else "")

    # 2. geometry
    t0 = time.time()
    # two-pass background: rough median -> track -> median excluding tracked vehicles
    bg0 = median_background(frames)
    bg, n_refined = bg0, 0
    for _ in range(2):          # each pass tracks better, so the next exclusion masks are better
        tracks_k = track_vehicles(frames, bg, TrackerConfig())
        bg, n_refined = refine_background(frames, tracks_k, bg0)
    _, rest_regions = event_background(frames)
    case["background"] = {"method": "two-pass temporal median excluding tracked vehicles",
                          "pixels_refined": n_refined, "objects_came_to_rest_or_left": rest_regions}
    timings["background"] = time.time() - t0
    # geometry from the plain median (markings are rarely covered most of the time); tracking from the refined one
    geo = estimate_road_geometry(frames, prior, GeometryConfig(seed=seed), background=bg0)
    timings["geometry"] = time.time() - t0
    case["geometry"] = geo.to_json()
    if not geo.ok:
        stage("geometry", "REJECT", "; ".join(geo.reasons))
        case.update({"status": "NOT_RECONSTRUCTABLE", "reasons": geo.reasons,
                     "identifiability": _identifiability(geo_ok=False)})
        case["timings_s"] = {k: round(v, 3) for k, v in timings.items()}
        return _json(case)
    stage("geometry", "OK", f"{geo.diagnostics.get('n_dashes')} dashes, roll {geo.roll_deg:+.2f} deg, "
                            f"calibrated over {geo.calibrated_x_range_m[1]-geo.calibrated_x_range_m[0]:.0f} m")

    # 3. tracking
    t0 = time.time()
    tracks = track_vehicles(frames, bg, TrackerConfig())
    timings["tracking"] = time.time() - t0

    out, internals = analyze_from_tracks(case, frames, fps, geo, tracks, prior, rng, n_mc, skeptic_checks, timings, t_start)
    if return_internals:
        internals["frames_shape"] = (h, w)
        return out, internals
    return out


def analyze_from_tracks(case, frames, fps, geo, tracks, prior, rng, n_mc, skeptic_checks, timings, t_start):
    """Stages 4-9 (kinematics -> verdict) from a list of tracks. Split out of
    analyze_video so the evaluation can feed ORACLE tracks (true contact points
    + pixel noise) through exactly the same reasoning code, separating
    detection/tracking errors from reasoning errors."""
    def stage(name, status, detail=""):
        case["stages"].append({"stage": name, "status": status, "detail": detail,
                               "t_s": round(time.time() - t_start, 2)})
    h, w = frames[0].shape[:2]
    # 4. kinematics per track (vehicles = tracks with >= 6 measurements spanning >= 3 m along the road)
    t0 = time.time()
    vehicles = []
    for tr in tracks:
        k = _kin_for(tr, fps, geo.H)
        if k is None or np.ptp(k["x"]) < 3.0:
            continue
        t_m, uv_m, boxes = _measured(tr, fps)
        last = tr["records"][-1]
        vehicles.append({"track": tr, "kin": k, "boxes": boxes, "uv": uv_m,
                         "ended_merged": last["status"] in ("MERGED", "OCCLUDED"),
                         "occluded_from_s": next((r["frame"] / fps for r in tr["records"] if r["status"] == "OCCLUDED"), None),
                         "merge_frames": [r["frame"] for r in tr["records"] if r["status"] in ("MERGED", "OCCLUDED")]})
    n_before = len(vehicles)
    vehicles = _stitch_tracklets(vehicles, fps, geo.H)
    timings["kinematics"] = time.time() - t0
    case["tracking"] = {"n_tracks_raw": len(tracks), "n_tracklets": n_before, "n_vehicles": len(vehicles),
                        "vehicles": [{"id": v["track"]["id"], "n_measured": v["track"]["n_measured"],
                                      "first_frame": v["track"]["records"][0]["frame"],
                                      "last_frame": v["track"]["records"][-1]["frame"],
                                      "ended_merged": v["ended_merged"]} for v in vehicles]}
    if not vehicles:
        stage("tracking", "REJECT", "no vehicle tracked over >= 3 m")
        case.update({"status": "NOT_RECONSTRUCTABLE", "reasons": ["no vehicle could be tracked over at least 3 m of road"],
                     "identifiability": _identifiability(geo_ok=True)})
        return _json(case), {"geo": geo, "tracks": tracks, "vehicles": locals().get("vehicles", []), "fps": fps, "pair": None}
    stage("tracking", "OK", f"{len(vehicles)} vehicle track(s)")

    # 5. pair + contact + onset
    pair = _find_pair(vehicles)
    case["vehicles"] = []
    for v in vehicles:
        k = v["kin"]
        case["vehicles"].append({"id": v["track"]["id"], "t_s": k["t"], "x_m": k["x"], "v_mps": k["v"], "a_mps2": k["a"],
                                 "y_m": k["y"], "sv_mps": k["sv"], "uv": v["uv"], "x_raw_m": k["x_raw"]})

    # 6+7. Monte Carlo + counterfactual
    t0 = time.time()
    if pair is None:
        stage("events", "NONE", "no same-lane following pair; speeds reported per vehicle")
        case["collision"] = None
        case["speeds"] = _speed_summaries(vehicles, geo, prior, fps, rng, n_mc)
        case["status"] = "RECONSTRUCTED_NO_INTERACTION"
    else:
        fol, lead, direction = pair
        fol = {**fol, "kin": _aligned(fol["kin"], direction), "direction": direction}
        lead = {**lead, "kin": _aligned(lead["kin"], direction), "direction": direction}
        pair = (fol, lead)
        L_f = FOLLOW_LENGTH_PRIOR[0]   # length between the two visible contacts: follower's (receding) or lead's (approaching)
        merged = fol["ended_merged"] or lead["ended_merged"]
        kl_ = lead["kin"]
        slow = np.where(kl_["v"] < 1.0)[0]
        t_stop = float(kl_["t"][slow[0]]) if len(slow) else None
        onset = hinge_onset(kl_["t"], kl_["x_raw"], kl_["sigma_x_meas"], t_data_max=t_stop)
        lead_brakes = onset is not None and onset["identified"]
        a_lead = -onset["decel_after_mps2"] if (onset and onset["decel_after_mps2"] > 1.5) else min(0.0, float(lead["kin"]["a"][-1]))
        braking_window = (lead["kin"]["t"][-1] - onset["t0_s"]) if lead_brakes else 0.0
        a_lead_rel_sigma = float(np.clip(0.15 / max(braking_window, 0.05), 0.08, 0.6))
        t_extrap, gap_at_lead_loss = _extrapolate_contact(fol["kin"], lead["kin"], L_f, a_lead)
        # direct evidence of impact: a sudden speed loss of the (still visible) follower
        kink = velocity_kink(fol["kin"]["t"], fol["kin"]["x_raw"], fol["kin"]["sigma_x_meas"],
                             t_lo=(t_extrap - 1.0) if t_extrap else lead["kin"]["t"][-1],
                             t_hi=(t_extrap + 1.0) if t_extrap else fol["kin"]["t"][-1])
        clip_end = (len(frames) - 1) / fps
        if t_extrap is not None and t_extrap > clip_end:
            t_extrap = None                      # would happen after the footage ends: not observed
        if kink is not None and kink["significant"]:
            # an impact needs the bumpers to be (nearly) touching; a hard brake onset can also look like a kink
            g_k = float(_lead_x(lead["kin"], a_lead, np.array([kink["t_s"]]))[0]
                        - (np.interp(kink["t_s"], fol["kin"]["t"], fol["kin"]["x"]) + L_f))
            kink["gap_at_kink_m"] = g_k
            if g_k > 1.0 + 2 * FOLLOW_LENGTH_PRIOR[1]:
                kink["significant"] = False
                kink["rejected_reason"] = f"reconstructed gap {g_k:.1f} m at the kink: a brake onset, not an impact"
        if kink is not None and kink["significant"]:
            t_contact, contact_source = kink["t_s"], "measured: sudden speed loss of the following vehicle"
        else:
            t_contact, contact_source = t_extrap, "extrapolated: lead hidden before contact"
            if t_extrap is not None:
                v_close = float(np.interp(t_extrap, fol["kin"]["t"], fol["kin"]["v"]) - _lead_v(lead["kin"], a_lead, t_extrap))
                if v_close < 2.0:
                    # vehicles came to rest (nearly) bumper to bumper: within the vehicle-length uncertainty,
                    # a contact cannot be established without an impact signature
                    t_contact, contact_source = None, (f"not established: closing speed {v_close:.1f} m/s at the "
                                                       f"extrapolated contact is within vehicle-length uncertainty "
                                                       f"and no impact signature was observed")
        lead_hidden_decel = None
        if t_contact is not None:
            # re-smooth both vehicles on PRE-IMPACT measurements only (post-contact motion must not leak
            # backwards through the smoother)
            kf_pre = _kin_for(fol["track"], fps, geo.H, t_max=t_contact)
            kl_pre = _kin_for(lead["track"], fps, geo.H, t_max=t_contact)
            if kf_pre is not None:
                fol["kin"] = _aligned(kf_pre, direction)
            if kl_pre is not None:
                lead["kin"] = _aligned(kl_pre, direction)
            if contact_source.startswith("measured") and lead["kin"]["t"][-1] < t_contact - 0.2:
                lead_hidden_decel = _lead_decel_from_impact(fol["kin"], lead["kin"], L_f, t_contact)
                if lead_hidden_decel is not None:
                    a_lead = -lead_hidden_decel
                    a_lead_rel_sigma = 0.25
        hazard = ({"source": "lead_brake_onset", "t_s": onset["t0_s"], "t_lo_s": onset["t0_lo_s"], "t_hi_s": onset["t0_hi_s"],
                   "lead_decel_after_mps2": onset["decel_after_mps2"],
                   "description": "the lead vehicle's measured brake onset"}
                  if lead_brakes else
                  {"source": "first_observation", "t_s": float(max(fol["kin"]["t"][0], lead["kin"]["t"][0])),
                   "hinge_fit": onset,
                   "description": "the first frame in which both vehicles are tracked (the lead vehicle's brake onset "
                                  "is not identifiable from its observed motion -- the true hazard onset is unknown)"})
        case["collision"] = {"detected": t_contact is not None, "t_contact_s": t_contact, "source": contact_source,
                             "traffic_direction": "away from camera" if direction > 0 else "towards camera",
                             "lead_hidden_decel_from_impact_mps2": lead_hidden_decel,
                             "t_contact_extrapolated_s": t_extrap, "impact_kink": kink,
                             "lead_decel_rel_sigma": a_lead_rel_sigma,
                             "lead_last_observed_s": float(lead["kin"]["t"][-1]),
                             "gap_when_lead_last_observed_m": gap_at_lead_loss,
                             "lead_decel_used_for_extrapolation_mps2": -a_lead,
                             "tracks_merged": merged, "follower_id": fol["track"]["id"], "lead_id": lead["track"]["id"],
                             "method": "constant-acceleration extrapolation of both vehicles from their last reliable "
                                       "positions to zero bumper gap (follower length prior "
                                       f"{FOLLOW_LENGTH_PRIOR[0]} +- {FOLLOW_LENGTH_PRIOR[1]} m)"}
        case["hazard_onset"] = hazard
        if t_contact is None:
            stage("events", "NO_CONTACT", "same-lane pair found, no contact extrapolated")
            case["status"] = "RECONSTRUCTED_NO_COLLISION"
            case["speeds"] = _speed_summaries(vehicles, geo, prior, fps, rng, n_mc)
        else:
            stage("events", "CONTACT", f"contact at t={t_contact:.2f}s; hazard onset ({hazard['source']}) t={hazard['t_s']:.2f}s")
            samples, speed_at_onset = _mc_samples(fol, lead, geo, prior, fps, hazard, rng, n_mc, a_lead,
                                                  a_lead_rel_sigma, t_contact if kink and kink["significant"] else None)
            cf = assess_counterfactual(samples, hazard_info=hazard)
            case["counterfactual"] = asdict(cf)
            case["speeds"] = _speed_summaries(vehicles, geo, prior, fps, rng, n_mc, ref_t=hazard["t_s"])
            case["follower_speed_at_hazard_onset"] = speed_at_onset
            impacts = [s.t_impact_s for s in samples if s.t_impact_s is not None]
            case["collision"]["t_contact_ci90_s"] = ([float(np.percentile(impacts, 5)), float(np.percentile(impacts, 95))]
                                                      if impacts else None)
            case["status"] = "RECONSTRUCTED_COLLISION"
    timings["events_mc_counterfactual"] = time.time() - t0

    # 8. cross-examination
    t0 = time.time()
    probe = np.vstack([v["uv"][[0, -1]] for v in vehicles])
    probe = probe[np.argsort(probe[:, 1])]
    t_cut = (case.get("collision") or {}).get("t_contact_s")
    kins_pre = []
    for v in vehicles:
        k = v["kin"]
        m = k["t"] < (t_cut - 0.3) if t_cut else np.ones(len(k["t"]), bool)
        if m.sum() >= 4:
            kins_pre.append({kk: (vv[m] if isinstance(vv, np.ndarray) and len(vv) == len(m) else vv) for kk, vv in k.items()})
    skep = cross_examine(frames, geo, prior, [v["boxes"] for v in vehicles], kins_pre,
                         probe[[0, -1]], enabled=skeptic_checks)
    timings["skeptic"] = time.time() - t0
    case["skeptic"] = skep
    blocking = [c for c in skep["disputed_checks"] if c in BLOCKING_CHECKS]
    if blocking:
        stage("skeptic", "DISPUTE", ", ".join(skep["disputed_checks"]))
        case["verdict"] = {"value": "INSUFFICIENT_EVIDENCE",
                           "reason": "the Skeptic disputed the reconstruction using independent evidence "
                                     f"({', '.join(blocking)}); the computed result is shown for review but no verdict is issued"}
        case["status_before_skeptic"] = case["status"]
        case["status"] = "DISPUTED"
    else:
        stage("skeptic", "PASS" if not skep["disputed"] else "ADVISORY", ", ".join(skep["disputed_checks"]) or "all checks passed")
        if case.get("counterfactual"):
            cfv = case["counterfactual"]["verdict"]
            if (case.get("hazard_onset") or {}).get("source") != "lead_brake_onset" and cfv in ("AVOIDABLE", "NOT_AVOIDABLE"):
                case["verdict"] = {"value": "ASSUMPTION_DEPENDENT",
                                   "reason": "the lead vehicle's brake onset was not observed, so the moment the hazard began is "
                                             "unknown; reaction-time budgets are only stated relative to the first tracked frame. "
                                             + case["counterfactual"]["narrative"]}
            else:
                case["verdict"] = {"value": cfv, "reason": case["counterfactual"]["narrative"]}
        else:
            case["verdict"] = {"value": "NO_COLLISION_ANALYSED", "reason": "no collision between tracked vehicles was found"}

    case["identifiability"] = _identifiability(geo_ok=True, hazard=case.get("hazard_onset"),
                                               follower=pair[0] if pair else None)
    timings["total"] = time.time() - t_start
    case["timings_s"] = {k: round(v, 3) for k, v in timings.items()}
    return _json(case), {"geo": geo, "tracks": tracks, "vehicles": vehicles, "fps": fps, "pair": pair}




def _stitch_tracklets(vehicles, fps, H, max_gap_s=1.5):
    """Link track fragments of the same vehicle (identity breaks under
    occlusion) in ROAD coordinates: B continues A if A's end state,
    extrapolated (deceleration allowed to bring it to rest), lands on B's
    first positions, laterally and in speed. Greedy, best match first."""
    changed = True
    while changed:
        changed = False
        best = None
        for a in vehicles:
            ka = a["kin"]
            for b in vehicles:
                if a is b:
                    continue
                kb = b["kin"]
                dt = kb["t"][0] - ka["t"][-1]
                if dt < -0.1 or dt > max_gap_s:
                    continue
                v0, acc = ka["v"][-1], ka["a"][-1]
                tau = max(dt, 0.0)
                if acc * v0 < 0 and abs(v0 / acc) < tau:          # comes to rest first
                    x_pred = ka["x"][-1] - 0.5 * v0 * v0 / acc
                    v_pred = 0.0
                else:
                    x_pred = ka["x"][-1] + v0 * tau + 0.5 * acc * tau * tau
                    v_pred = v0 + acc * tau
                xb0 = float(np.median(kb["x_raw"][:3]))
                tol_x = 1.5 + 0.3 * abs(v0) * tau + 2 * float(kb["sigma_x_meas"][0])
                dx = abs(x_pred - xb0)
                dy = abs(ka["y"][-1] - float(np.median(kb["y_raw"][:3])))
                dv = abs(v_pred - kb["v"][min(3, len(kb["v"]) - 1)])
                if dx < tol_x and dy < 1.0 and dv < 5.0:
                    score = dx / tol_x + dy + dv / 5.0
                    if best is None or score < best[0]:
                        best = (score, a, b)
        if best is not None:
            _, a, b = best
            frames_a = {r["frame"] for r in a["track"]["records"]}
            recs = a["track"]["records"] + [r for r in b["track"]["records"] if r["frame"] not in frames_a]
            recs.sort(key=lambda r: r["frame"])
            a["track"] = {**a["track"], "records": recs,
                          "n_measured": a["track"]["n_measured"] + b["track"]["n_measured"],
                          "stitched_from": a["track"].get("stitched_from", [a["track"]["id"]]) + [b["track"]["id"]]}
            k = _kin_for(a["track"], fps, H)
            if k is not None:
                a["kin"] = k
                t_m, uv_m, boxes = _measured(a["track"], fps)
                a["uv"], a["boxes"] = uv_m, boxes
                a["ended_merged"] = b["ended_merged"]
                vehicles.remove(b)
                changed = True
    return vehicles


def _find_pair(vehicles):
    """Same-lane pair (|dy| < 1.6 m over the overlap), moving the same way,
    one behind the other in the direction of travel and closing. Returns
    (follower, lead, direction) with direction = +1 if traffic moves away
    from the camera (visible contacts are REAR bumpers) or -1 if towards it
    (visible contacts are FRONT bumpers). Picks the pair with the smallest gap."""
    best = None
    for i, a in enumerate(vehicles):
        for j, b in enumerate(vehicles):
            if i == j:
                continue
            ka, kb = a["kin"], b["kin"]
            t_lo, t_hi = max(ka["t"][0], kb["t"][0]), min(ka["t"][-1], kb["t"][-1])
            if t_hi - t_lo < 0.5:
                continue
            da, db = np.sign(np.median(ka["v"])), np.sign(np.median(kb["v"]))
            if da == 0 or da != db:
                continue
            tt = np.linspace(t_lo, t_hi, 50)
            xa, xb = da * np.interp(tt, ka["t"], ka["x"]), da * np.interp(tt, kb["t"], kb["x"])
            ya, yb = np.interp(tt, ka["t"], ka["y"]), np.interp(tt, kb["t"], kb["y"])
            if np.median(np.abs(ya - yb)) > 1.6 or not np.all(xb > xa):
                continue   # need a = follower (behind in the direction of travel), b = lead
            gap = xb - xa
            if gap[0] - gap[-1] <= 0.5:
                continue
            if best is None or gap.min() < best[0]:
                best = (gap.min(), a, b, int(da))
    return (best[1], best[2], best[3]) if best else None


def _aligned(kin, direction):
    """Kinematics in 'direction of travel' coordinates (x increases along travel)."""
    if direction > 0:
        return kin
    out = dict(kin)
    for key in ("x", "v", "a", "x_raw"):
        out[key] = -np.asarray(kin[key])
    return out


def _speed_summaries(vehicles, geo, prior, fps, rng, n_mc, ref_t=None):
    out = []
    Hs = geo.H_samples or [geo.H]
    for v in vehicles:
        k = v["kin"]
        t_ref = k["t"][len(k["t"]) // 3] if ref_t is None or not (k["t"][0] <= ref_t <= k["t"][-1]) else ref_t
        draws = []
        for m in range(n_mc):
            Hm = Hs[rng.integers(len(Hs))]
            sx = 1.0 + rng.normal(0, prior.cycle_rel_sigma)
            km = _kin_for(v["track"], fps, Hm, sx=sx)
            if km is None:
                continue
            vm = float(np.interp(t_ref, km["t"], km["v"]))
            sv = float(np.interp(t_ref, km["t"], km["sv"]))
            draws.append(vm + rng.normal(0, sv))
        v_point = float(np.interp(t_ref, k["t"], k["v"]))
        out.append({"vehicle_id": v["track"]["id"], "t_ref_s": float(t_ref), "speed_mps": v_point,
                    "ci90_mps": [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))] if draws else None,
                    "uncertainty_budget_rel_sigma": {
                        "marking_prior": prior.cycle_rel_sigma,
                        "geometry_and_tracking": float(np.std(draws) / max(abs(v_point), 0.5)) if draws else None}})
    return out


def _mc_samples(fol, lead, geo, prior, fps, hazard, rng, n_mc, a_lead, a_lead_rel_sigma=0.1, t_contact_measured=None):
    Hs = geo.H_samples or [geo.H]
    samples, v_onsets = [], []
    for m in range(n_mc):
        Hm = Hs[rng.integers(len(Hs))]
        sx = 1.0 + rng.normal(0, prior.cycle_rel_sigma)
        kf = _kin_for(fol["track"], fps, Hm, sx=sx, t_max=t_contact_measured)
        kl = _kin_for(lead["track"], fps, Hm, sx=sx, t_max=t_contact_measured)
        if kf is None or kl is None:
            continue
        kf, kl = _aligned(kf, fol.get("direction", 1)), _aligned(kl, fol.get("direction", 1))
        L_f = max(3.5, rng.normal(*FOLLOW_LENGTH_PRIOR))
        if hazard["source"] == "lead_brake_onset":
            th = rng.uniform(hazard["t_lo_s"], hazard["t_hi_s"] + 1e-6)
        else:
            th = hazard["t_s"]
        a_l = min(0.0, a_lead * (1.0 + rng.normal(0, a_lead_rel_sigma))) if a_lead < 0 else a_lead
        if t_contact_measured is not None:
            tc = t_contact_measured + rng.normal(0, 1.0 / fps)
            if kl["t"][-1] < tc - 0.2:
                dh = _lead_decel_from_impact(kf, kl, L_f, tc)
                if dh is not None:
                    a_l = -dh
        else:
            tc, _ = _extrapolate_contact(kf, kl, L_f, a_l)
        dv = rng.normal(0, float(np.interp(th, kf["t"], kf["sv"])))
        t = np.union1d(kf["t"], kl["t"])
        x_l = _lead_x(kl, a_l, t)
        v_l = np.interp(t, kl["t"], kl["v"])
        beyond = t > kl["t"][-1]
        v_l[beyond] = np.maximum(0.0, kl["v"][-1] + a_l * (t[beyond] - kl["t"][-1]))
        a_end = a_l
        kf = {**kf, "x": np.interp(t, kf["t"], kf["x"]), "v": np.interp(t, kf["t"], kf["v"]), "t": t}
        vf = kf["v"] + dv
        # truncate to before contact: the counterfactual only uses pre-impact motion
        if tc is not None:
            keep = t <= tc
            t, x_l, v_l, vf = t[keep], x_l[keep], v_l[keep], vf[keep]
            kf = {**kf, "x": kf["x"][keep], "t": t}
        samples.append(TrajectorySample(t=t, x_follow=kf["x"], v_follow=vf, x_lead=x_l, v_lead=v_l,
                                        a_lead_end=a_end, follow_length_m=L_f, hazard_onset_s=th, t_impact_s=tc))
        v_onsets.append(float(np.interp(th, t, vf)))
    summ = None
    if v_onsets:
        summ = {"median_mps": float(np.median(v_onsets)),
                "ci90_mps": [float(np.percentile(v_onsets, 5)), float(np.percentile(v_onsets, 95))]}
    return samples, summ


def _identifiability(geo_ok, hazard=None, follower=None):
    rows = [
        {"quantity": "Along-road displacement and speed", "status": "MEASURED_GIVEN_PRIOR" if geo_ok else "NOT_IDENTIFIABLE",
         "basis": "dash pattern seen in the image; metric scale = assumed marking cycle length (stated prior, +-3 %)"},
        {"quantity": "Lateral position / lane offset", "status": "MEASURED_GIVEN_PRIOR" if geo_ok else "NOT_IDENTIFIABLE",
         "basis": "edge lines; metric scale = assumed edge-to-centre distance (stated prior, +-5 %)"},
        {"quantity": "Absolute position along the road", "status": "NOT_IDENTIFIABLE",
         "basis": "the image has no landmark of known absolute position; only relative positions are reported"},
        {"quantity": "Camera roll", "status": "MEASURED" if geo_ok else "NOT_IDENTIFIABLE",
         "basis": "horizon from the harmonic conjugate of the three road lines"},
        {"quantity": "Lead vehicle brake onset", "status": ("MEASURED" if hazard and hazard.get("source") == "lead_brake_onset"
                                                            else "NOT_OBSERVED"),
         "basis": "hinge fit to the lead vehicle's raw positions"},
        {"quantity": "Contact time", "status": "MEASURED_BY_EXTRAPOLATION",
         "basis": "the final metres are hidden by occlusion; extrapolated from the last reliable positions"},
        {"quantity": "Follower vehicle length", "status": "ASSUMED", "basis": f"prior {FOLLOW_LENGTH_PRIOR[0]} +- {FOLLOW_LENGTH_PRIOR[1]} m"},
        {"quantity": "Driver reaction time", "status": "NOT_IDENTIFIABLE",
         "basis": "not visible in video -- evaluated over 0.75-2.0 s (envelope)"},
        {"quantity": "Achievable braking deceleration (tyre-road friction)", "status": "NOT_IDENTIFIABLE",
         "basis": "evaluated over 5-8 m/s^2 (envelope); an observed hard stop is a lower bound only"},
    ]
    if follower is not None:
        peak = float(-np.min(follower["kin"]["a"]))
        if peak > 2.0:
            rows[-1]["basis"] += f"; the follower was itself observed decelerating at up to {peak:.1f} m/s^2"
    return rows
