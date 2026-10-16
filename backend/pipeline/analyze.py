"""
Odolite production pipeline (single entry point used by the evaluation
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

import copy as _copy
import hashlib
import os
import time
from dataclasses import asdict

import cv2
import numpy as np

from backend.geometry.road_geometry import (
    RoadPrior, GeometryConfig, estimate_road_geometry, image_to_road, road_to_image, median_background, event_background,
)
from cv.tracking.mot import (TrackerConfig, track_vehicles, bottom_centre, contact_point, refine_background,
                             resolve_changed_regions)
from backend.reconstruction.kinematics import road_kinematics, velocity_kink
from backend.causality.counterfactual import TrajectorySample, assess as assess_counterfactual, hinge_onset
from backend.verification.skeptic import cross_examine

PIPELINE_VERSION = "odolite-2.2"   # = carai-2.2 (the frozen v2.2 code under the working name)
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
    """Fraction of reliably tracked corner features that do not move between
    frames gap_s apart. Features are found on a lightly blurred image and kept
    only if forward-backward Lucas-Kanade tracking is consistent (< 0.5 px), so
    sensor noise on a textureless scene cannot pose as camera motion. Returns
    (static, fraction); fraction None if too few reliable features (then the
    camera is treated as static and later stages decide reconstructability)."""
    n = len(frames)
    gap = max(1, int(gap_s * fps))
    if n <= gap:
        return True, 1.0
    fracs = []
    for i in np.linspace(0, n - 1 - gap, n_pairs).astype(int):
        g0 = cv2.GaussianBlur(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), (5, 5), 1.0)
        g1 = cv2.GaussianBlur(cv2.cvtColor(frames[i + gap], cv2.COLOR_BGR2GRAY), (5, 5), 1.0)
        pts = cv2.goodFeaturesToTrack(g0, 300, 0.02, 8)
        if pts is None or len(pts) < 10:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None)
        p0b, st_b, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None)
        fb = np.linalg.norm((p0b - pts).reshape(-1, 2), axis=1)
        ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb < 0.5)
        if ok.sum() < 10:
            continue
        mv = np.linalg.norm((p1 - pts).reshape(-1, 2)[ok], axis=1)
        fracs.append(float(np.mean(mv < 1.0)))
    if not fracs:
        return True, None
    f = float(np.median(fracs))
    # measured: moving dashcams 0.00-0.09 (a visible bonnet is static); static sim scenes with traffic 0.66-1.0
    return f > 0.4, f


# ------------------------------------------------------------- utilities --

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


OCCLUSION_GUARD_FRAMES = 3
WORST_CASE_DECEL_MPS2 = 8.0          # upper end of the braking envelope (v2.1 no-contact check)
# background before tracking: "legacy" = whole-clip median + first-half fallback in the refinement (default);
# "resolve3" / "resolve5" = segment medians resolved region by region (cv.tracking.mot.resolve_changed_regions).
# Dev-split ablation, end to end (evaluation/results/dev_history/ablation_bg_*.jsonl): legacy 7/30 correct,
# 1 harmful; resolve3 7/30, 2 harmful; resolve5 8/30, 4 harmful -> legacy kept. The env var exists only for
# that ablation.
BACKGROUND_MODE = os.environ.get("ODOLITE_BG_MODE", "legacy")


def _measured(track, fps):
    """Measured ground contacts of a track. A vehicle's contact in the last
    frames before it is hidden behind another (and the first frames after it
    re-emerges) is already partly covered -- the two silhouettes touch before
    the tracker can call them merged -- so those few MEASURED records are not
    used (OCCLUSION_GUARD_FRAMES on each side of an OCCLUDED run)."""
    allr = track["records"]
    occ = np.array([r["status"] == "OCCLUDED" for r in allr], bool)
    near_occ = np.zeros(len(allr), bool)
    for i in np.where(occ)[0]:
        near_occ[max(0, i - OCCLUSION_GUARD_FRAMES): i + OCCLUSION_GUARD_FRAMES + 1] = True
    recs = [r for r, n in zip(allr, near_occ)
            if r["status"] == "MEASURED_OCCLUDING" or (r["status"] == "MEASURED" and not n)]
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


def _follow_state(kin_f, tq, a_follow=None):
    """Follower position and speed: measured, then (if its ground contact is
    hidden -- in traffic approaching the camera the FOLLOWER is the occluded
    one) extrapolated from its last measured state at constant acceleration
    a_follow (default: its last measured acceleration if braking, else 0),
    to rest if decelerating."""
    tq = np.atleast_1d(np.asarray(tq, float))
    x = np.interp(tq, kin_f["t"], kin_f["x"])
    v = np.interp(tq, kin_f["t"], kin_f["v"])
    after = tq > kin_f["t"][-1]
    if np.any(after):
        v0 = max(0.0, float(kin_f["v"][-1]))
        a0 = min(0.0, float(kin_f["a"][-1])) if a_follow is None else float(a_follow)
        d = tq[after] - kin_f["t"][-1]
        ts = np.minimum(d, v0 / -a0) if a0 < -1e-6 else d
        x[after] = kin_f["x"][-1] + v0 * ts + 0.5 * a0 * ts ** 2
        v[after] = np.maximum(0.0, v0 + a0 * ts)
    return x, v


def _extrapolate_contact(kin_f, kin_l, L_f, a_lead, horizon_s=0.2, dt=0.005, a_follow=None, t_until=None):
    """First time the bumper gap x_lead - (x_follow + L_f) reaches 0. Whichever
    vehicle is hidden first is extrapolated from its last measured state (the
    lead with a_lead, the follower with a_follow); the search runs until
    horizon_s after the LATER of the two last observations (or until t_until).
    Returns (t, gap_at_last_lead_obs) or (None, ...)."""
    t_lo = max(kin_f["t"][0], kin_l["t"][0])
    t_hi = t_until if t_until is not None else max(kin_f["t"][-1], kin_l["t"][-1]) + horizon_s
    tq = np.arange(t_lo, t_hi, dt)
    if len(tq) == 0:
        return None, None
    xf, _ = _follow_state(kin_f, tq, a_follow)
    gap = _lead_x(kin_l, a_lead, tq) - (xf + L_f)
    hit = np.where(gap <= 0)[0]
    gap_last = float(np.interp(kin_l["t"][-1], tq, gap)) if kin_l["t"][-1] >= t_lo else None
    if len(hit) == 0:
        return None, gap_last
    return float(tq[hit[0]]), gap_last


def _follow_accel_from_impact(kin_f, kin_l, L_f, t_c):
    """Mirror of _lead_decel_from_impact for a hidden FOLLOWER: the constant
    acceleration that brings its front (x_follow + L_f) to the lead's rear at
    the measured impact t_c. Bounded to physical values; None if the
    follower was visible until (nearly) t_c."""
    t1 = kin_f["t"][-1]
    tau = t_c - t1
    if tau <= 0.2:
        return None
    x1, v1 = float(kin_f["x"][-1]), max(0.0, float(kin_f["v"][-1]))
    x_c = float(np.interp(t_c, kin_l["t"], kin_l["x"])) - L_f
    a = 2.0 * (x_c - x1 - v1 * tau) / tau ** 2
    return float(np.clip(a, -10.0, 3.0))


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
                  return_internals=False, fps_override=None, frame_range=None, focal_px=None, preloaded=None):
    """fps_override: the true frame rate when the user supplies it (container metadata is often wrong).
    frame_range: (first, last) frame indices -- analyse only that part of the clip (used by the
    investigator for the segments either side of a camera bump); times are then relative to `first`."""
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
        # preloaded: (fps, frames, info) already decoded from this file (the investigator re-analyses parts of
        # a clip without decoding it again, which would double peak memory)
        fps, frames, info = (preloaded[0], preloaded[1], dict(preloaded[2])) if preloaded else ingest(path, max_seconds)
    except ValueError as e:
        case.update({"status": "INVALID_INPUT", "reasons": [str(e)]})
        stage("ingest", "FAIL", str(e))
        return (_json(case), None) if return_internals else _json(case)
    case["input"].update(info)
    if fps_override:
        case["input"]["fps_container"] = fps
        fps = float(fps_override)
        case["input"]["fps"] = fps
        case["input"]["fps_source"] = "supplied by the user"
    else:
        case["input"]["fps_source"] = "container metadata (unverified)"
    all_frames = frames
    if frame_range is not None:
        a, b = int(frame_range[0]), int(min(frame_range[1], len(frames)))
        frames = frames[a:b]
        case["input"]["frame_range"] = [a, b]
        case["input"]["t_offset_s"] = a / fps
    if len(frames) < 15:
        case.update({"status": "INVALID_INPUT", "reasons": [f"only {len(frames)} frames decoded (need >= 15)"]})
        stage("ingest", "FAIL", "too few frames")
        return (_json(case), None) if return_internals else _json(case)
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
            from backend.dashcam.ego_speed import EgoSpeedConfig, estimate_ego_speed, MarkingPrior
            mp = MarkingPrior("from case prior", prior.cycle_m, prior.cycle_rel_sigma)
            case["dashcam"] = estimate_ego_speed(path, mp, max_frames=len(frames), fps_override=fps_override,
                                                 cfg=EgoSpeedConfig(focal_px=focal_px))
        except Exception as e:  # noqa: BLE001
            case["dashcam_error"] = str(e)
        # lead vehicle: range, closing speed, time-to-collision (needs the camera's focal length: both of its
        # range rulers -- the ground plane and the vehicle width -- are in pixels per metre at the focal length)
        if focal_px:
            try:
                from backend.dashcam.lead_vehicle import LeadConfig, analyze_lead
                case["dashcam_lead"] = analyze_lead(path, fps, LeadConfig(focal_px=float(focal_px)), max_frames=len(frames))
            except Exception as e:  # noqa: BLE001
                case["dashcam_lead"] = {"status": "ERROR", "reason": str(e)[:300]}
        else:
            case["dashcam_lead"] = {"status": "NEEDS_INPUT",
                                    "reason": "the lead-vehicle range needs the camera's focal length in pixels "
                                              "(device specification or calibration)"}
        return (_json(case), None) if return_internals else _json(case)
    stage("camera_motion", "STATIC", f"{frac:.0%} of features static" if frac is not None else "")

    # 2. geometry
    t0 = time.time()
    # two-pass background: rough median -> track -> median excluding tracked vehicles
    bg0 = median_background(frames)
    decisions = []
    if BACKGROUND_MODE.startswith("resolve"):
        # start/end disagreements (vehicles at rest after the event, or crawling through at the start) are
        # resolved region by region before any tracking, instead of by a first-half rule afterwards
        bg_base, decisions = resolve_changed_regions(frames, bg0, n_segments=int(BACKGROUND_MODE[len("resolve"):]))
    else:
        bg_base = bg0
    bg, n_refined = bg_base, 0
    for it in range(2):         # each pass tracks better, so the next exclusion masks are better
        stride = 2                            # these passes only have to find where vehicles are
        tracks_k = track_vehicles(frames[::stride], bg, TrackerConfig())
        if stride > 1:
            for tr in tracks_k:
                for r in tr["records"]:
                    r["frame"] *= stride
        bg, n_refined = refine_background(frames, tracks_k, bg_base,
                                          use_first_half_where_changed=(BACKGROUND_MODE == "legacy"))
    _, rest_regions = event_background(frames)
    case["background"] = {"method": ("segment medians resolved by boundary contrast, then " if BACKGROUND_MODE.startswith("resolve") else "")
                                    + "two-pass temporal median excluding tracked vehicles",
                          "resolved_regions": decisions,
                          "pixels_refined": n_refined, "objects_came_to_rest_or_left": rest_regions}
    timings["background"] = time.time() - t0
    # geometry from the plain median (markings are rarely covered most of the time); tracking from the refined one
    geo = estimate_road_geometry(frames, prior, GeometryConfig(seed=seed), background=bg0)
    if not geo.ok and len(frames) >= 60:
        # a camera that moved mid-clip blurs the whole-clip median (doubled lines). Try the first half alone:
        # if that gives a geometry, the Skeptic's temporal-stability check decides whether the rest agrees.
        geo_h = estimate_road_geometry(frames, prior, GeometryConfig(seed=seed),
                                       background=median_background(frames, 48, (0.0, 0.5)))
        if geo_h.ok:
            geo_h.diagnostics["background"] = ("first half of the clip (the whole-clip background did not support a "
                                               "geometry: " + "; ".join(geo.reasons) + ")")
            geo = geo_h
    timings["geometry"] = time.time() - t0
    case["geometry"] = geo.to_json()
    if not geo.ok:
        stage("geometry", "REJECT", "; ".join(geo.reasons))
        case.update({"status": "NOT_RECONSTRUCTABLE", "reasons": geo.reasons,
                     "identifiability": _identifiability(geo_ok=False)})
        case["timings_s"] = {k: round(v, 3) for k, v in timings.items()}
        return (_json(case), None) if return_internals else _json(case)
    stage("geometry", "OK", f"{geo.diagnostics.get('n_dashes')} dashes, roll {geo.roll_deg:+.2f} deg, "
                            f"calibrated over {geo.calibrated_x_range_m[1]-geo.calibrated_x_range_m[0]:.0f} m")

    # 3. tracking
    t0 = time.time()
    tracks = track_vehicles(frames, bg, TrackerConfig())
    timings["tracking"] = time.time() - t0

    case_pre = _copy.deepcopy(case) if return_internals else None
    out, internals = analyze_from_tracks(case, frames, fps, geo, tracks, prior, rng, n_mc, skeptic_checks, timings, t_start)
    if return_internals:
        internals["frames_shape"] = (h, w)
        internals["ingest"] = (case["input"].get("fps_container", case["input"]["fps"]), all_frames, info)
        internals.update({"frames": frames, "case_pre": case_pre, "prior": prior, "seed": seed, "n_mc": n_mc,
                          "skeptic_checks": skeptic_checks})
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
        a_lead = (-onset["decel_after_mps2"] if (onset and 1.5 < onset["decel_after_mps2"] <= 12.0)
                  else max(-12.0, min(0.0, float(lead["kin"]["a"][-1]))))
        braking_window = (lead["kin"]["t"][-1] - onset["t0_s"]) if lead_brakes else 0.0
        a_lead_rel_sigma = float(np.clip(0.15 / max(braking_window, 0.05), 0.08, 0.6))
        t_extrap, gap_at_lead_loss = _extrapolate_contact(fol["kin"], lead["kin"], L_f, a_lead)
        clip_end = (len(frames) - 1) / fps
        hidden = "lead" if lead["kin"]["t"][-1] < fol["kin"]["t"][-1] else "follower"
        # only the vehicle FARTHER from the camera can be hidden behind the other (the lead when traffic
        # recedes, the follower when it approaches). If the nearer one's track ends first, it was LOST by the
        # tracker, and extrapolating it across the approach would invent the event.
        far = "lead" if direction > 0 else "follower"
        t_end_f, t_end_l = fol["kin"]["t"][-1], lead["kin"]["t"][-1]
        near_lost_at = None
        if hidden != far and abs(t_end_f - t_end_l) > 0.3:
            near_lost_at = min(t_end_f, t_end_l)
        # direct evidence of impact: a sudden speed change of the vehicle that stays visible -- the follower
        # loses speed (traffic receding: the follower occludes the lead) or the lead is pushed forward
        # (traffic approaching: the lead occludes the follower)
        win = lambda k: ((t_extrap - 1.0) if t_extrap else max(fol["kin"]["t"][0], lead["kin"]["t"][0]),
                         (t_extrap + 1.0) if t_extrap else k["t"][-1])
        kink = velocity_kink(fol["kin"]["t"], fol["kin"]["x_raw"], fol["kin"]["sigma_x_meas"],
                             t_lo=win(fol["kin"])[0], t_hi=win(fol["kin"])[1])
        if kink is not None:
            kink["vehicle"] = "follower"
        # v2.2: a "pushed lead" is evidence only while the lead stays in view through the contact -- traffic
        # approaching (the lead is the nearer vehicle). Receding traffic hides the lead behind the follower, and
        # the last positions of a vehicle being covered move as its silhouette shrinks: not an impact.
        lead_kink_ok = direction < 0 or (hidden != "lead" and lead.get("occluded_from_s") is None)
        if not (kink and kink["significant"]) and lead_kink_ok:
            lk = velocity_kink(lead["kin"]["t"], -np.asarray(lead["kin"]["x_raw"]), lead["kin"]["sigma_x_meas"],
                               t_lo=win(lead["kin"])[0], t_hi=win(lead["kin"])[1])
            if lk is not None and lk["significant"]:
                kink = {**lk, "dv_mps": -lk["dv_mps"], "vehicle": "lead"}
        if not (kink and kink["significant"]):
            # both vehicles visible through the contact: at impact the CLOSING speed drops to ~0 in one frame
            # (the pair then moves together) -- the gap series carries both vehicles' speed changes
            kf_, kl_g = fol["kin"], lead["kin"]
            tf, tl = np.round(kf_["t"] * fps).astype(int), np.round(kl_g["t"] * fps).astype(int)
            common, i_f, i_l = np.intersect1d(tf, tl, return_indices=True)
            if len(common) >= 12:
                closing = kf_["x_raw"][i_f] - kl_g["x_raw"][i_l]           # = -(gap + L): its slope is the closing speed
                sig = np.hypot(kf_["sigma_x_meas"][i_f], kl_g["sigma_x_meas"][i_l])
                gk = velocity_kink(common / fps, closing, sig, t_lo=win(kf_)[0], t_hi=win(kf_)[1])
                if gk is not None and gk["significant"]:
                    kink = {**gk, "vehicle": "pair"}
        if t_extrap is not None and t_extrap > clip_end:
            t_extrap = None                      # would happen after the footage ends: not observed
        contact_length_note = None
        if t_extrap is not None:
            # the extrapolated contact must survive the follower-length uncertainty: with a follower 2 sigma
            # shorter than the prior, contact comes later -- if that is after the footage ends (or never),
            # the footage does not establish that the vehicles touched
            t_short, _ = _extrapolate_contact(fol["kin"], lead["kin"], L_f - 2 * FOLLOW_LENGTH_PRIOR[1], a_lead)
            if t_short is None or t_short > clip_end:
                contact_length_note = (f"not established: contact at t={t_extrap:.2f}s with the {L_f} m length prior, but a "
                                       f"vehicle {2 * FOLLOW_LENGTH_PRIOR[1]:.1f} m shorter would not reach the other "
                                       "before the footage ends")
        if kink is not None and kink["significant"]:
            # an impact needs the bumpers to be (nearly) touching; a hard brake onset can also look like a kink
            xf_k, _ = _follow_state(fol["kin"], [kink["t_s"]])
            g_k = float(_lead_x(lead["kin"], a_lead, np.array([kink["t_s"]]))[0] - (xf_k[0] + L_f))
            kink["gap_at_kink_m"] = g_k
            if g_k > 1.0 + 2 * FOLLOW_LENGTH_PRIOR[1]:
                kink["significant"] = False
                kink["rejected_reason"] = f"reconstructed gap {g_k:.1f} m at the kink: a brake onset, not an impact"
        if kink is not None and kink["significant"]:
            t_contact = kink["t_s"]
            contact_source = {"follower": "measured: sudden speed loss of the following vehicle",
                              "lead": "measured: the lead vehicle was pushed forward",
                              "pair": "measured: the closing speed dropped to zero within a frame (both vehicles visible)"
                              }[kink["vehicle"]]
        else:
            t_contact, contact_source = t_extrap, f"extrapolated: {hidden} hidden before contact"
            if near_lost_at is not None and (t_extrap is None or t_extrap > near_lost_at + 0.2):
                t_contact, contact_source = None, (f"not established: the {hidden}, the vehicle nearer the camera, "
                                                   f"stopped being tracked at t={near_lost_at:.2f}s while the other was "
                                                   "still in view, so the approach was not observed")
            elif contact_length_note is not None:
                t_contact, contact_source = None, contact_length_note
            elif t_extrap is not None:
                _, vf_c = _follow_state(fol["kin"], [t_extrap])
                v_close = float(vf_c[0] - _lead_v(lead["kin"], a_lead, t_extrap))
                if v_close < 2.0:
                    # vehicles came to rest (nearly) bumper to bumper: within the vehicle-length uncertainty,
                    # a contact cannot be established without an impact signature
                    t_contact, contact_source = None, (f"not established: closing speed {v_close:.1f} m/s at the "
                                                       f"extrapolated contact is within vehicle-length uncertainty "
                                                       f"and no impact signature was observed")
            elif t_extrap is None:
                # "no contact" is a claim too: it needs the footage to exclude contact. A vehicle 2 sigma longer
                # than the prior, or vehicles lost from view while still closing, leave it open.
                t_long, _ = _extrapolate_contact(fol["kin"], lead["kin"], L_f + 2 * FOLLOW_LENGTH_PRIOR[1], a_lead,
                                                 t_until=clip_end)
                t_end = max(fol["kin"]["t"][-1], lead["kin"]["t"][-1])
                _, vf_e = _follow_state(fol["kin"], [t_end])
                closing_at_end = float(vf_e[0] - _lead_v(lead["kin"], a_lead, t_end)) > 0.5
                # v2.1 (post-freeze): while a vehicle is hidden its motion is unobserved. "No contact" must also hold
                # if the hidden lead braked as hard as the envelope allows or the hidden follower did not brake.
                t_worst = None
                if abs(fol["kin"]["t"][-1] - lead["kin"]["t"][-1]) > 0.2:
                    a_l_w = min(a_lead, -WORST_CASE_DECEL_MPS2) if hidden == "lead" else a_lead
                    a_f_w = 0.0 if hidden == "follower" else None
                    t_worst, _ = _extrapolate_contact(fol["kin"], lead["kin"], L_f + 2 * FOLLOW_LENGTH_PRIOR[1], a_l_w,
                                                      a_follow=a_f_w, t_until=clip_end)
                if t_worst is not None and t_worst <= clip_end:
                    contact_source = (f"not established: the {hidden} was hidden from view; had it "
                                      + ("braked at up to 8 m/s^2" if hidden == "lead" else "not braked")
                                      + f" while hidden, the vehicles would have touched at t={t_worst:.2f}s; "
                                      "the footage cannot exclude contact")
                elif t_long is not None and t_long <= clip_end:
                    contact_source = (f"not established: no contact with the {L_f} m length prior, but a vehicle "
                                      f"{2 * FOLLOW_LENGTH_PRIOR[1]:.1f} m longer would touch at t={t_long:.2f}s; "
                                      "the footage cannot exclude contact")
                elif t_end < clip_end - 0.5 and closing_at_end:
                    contact_source = (f"not established: both vehicles were out of view after t={t_end:.2f}s while "
                                      "still closing; the footage does not show whether they touched")
                else:
                    contact_source = "no contact: the reconstructed gap stays open while the vehicles are in view"
        res_m_per_row = None
        try:
            from backend.reconstruction.kinematics import _jacobian
            # v2.2: at the event (the contact, else the closest approach while both are measured), not at the
            # follower's last position
            t_mf = _measured(fol["track"], fps)[0]
            if t_contact is not None:
                t_ev = t_contact
            else:
                tc_ = fol["kin"]["t"][(fol["kin"]["t"] >= lead["kin"]["t"][0]) & (fol["kin"]["t"] <= lead["kin"]["t"][-1])]
                t_ev = (float(tc_[np.argmin(np.interp(tc_, lead["kin"]["t"], lead["kin"]["x"])
                                            - np.interp(tc_, fol["kin"]["t"], fol["kin"]["x"]))])
                        if len(tc_) else float(t_mf[-1]))
            k_ev = int(np.argmin(np.abs(t_mf - t_ev)))
            res_m_per_row = float(abs(_jacobian(geo.H, fol["uv"][k_ev])[0, 1]))
        except Exception:  # noqa: BLE001 -- explanatory only
            pass
        if (res_m_per_row is not None and res_m_per_row > 2 * FOLLOW_LENGTH_PRIOR[1]
                and not contact_source.startswith("not established")):
            # v2.1 (post-freeze): at this range pixel quantisation, not the length prior, limits the measured gap,
            # so neither a contact (or its time) nor its absence is established
            t_contact = None
            contact_source = (f"not established: the vehicles are too far from the camera for the gap between them "
                              f"to be measured to within the vehicle-length uncertainty "
                              f"({2 * FOLLOW_LENGTH_PRIOR[1]:.1f} m)")
        if contact_source.startswith("not established") and res_m_per_row is not None:
            contact_source += (f" (at the vehicles' position one image row spans {res_m_per_row:.2f} m of road, "
                               "which limits how precisely the gap can be measured)")
        lead_hidden_decel = follow_hidden_accel = None
        if t_contact is not None:
            # re-smooth both vehicles on PRE-IMPACT measurements only (post-contact motion must not leak
            # backwards through the smoother)
            kf_pre = _kin_for(fol["track"], fps, geo.H, t_max=t_contact)
            kl_pre = _kin_for(lead["track"], fps, geo.H, t_max=t_contact)
            if kf_pre is not None:
                fol["kin"] = _aligned(kf_pre, direction)
            if kl_pre is not None:
                lead["kin"] = _aligned(kl_pre, direction)
            # the lead's brake onset must be fitted on pre-impact motion only (a pushed lead is not braking)
            if lead["kin"]["t"][-1] < kl_["t"][-1]:
                t_dm = min(t_stop, t_contact) if t_stop is not None else t_contact
                onset = hinge_onset(lead["kin"]["t"], lead["kin"]["x_raw"], lead["kin"]["sigma_x_meas"], t_data_max=t_dm)
                lead_brakes = onset is not None and onset["identified"]
                if onset and 1.5 < onset["decel_after_mps2"] <= 12.0:
                    a_lead = -onset["decel_after_mps2"]
            if contact_source.startswith("measured") and lead["kin"]["t"][-1] < t_contact - 0.2:
                lead_hidden_decel = _lead_decel_from_impact(fol["kin"], lead["kin"], L_f, t_contact)
                if lead_hidden_decel is not None:
                    a_lead = -lead_hidden_decel
                    a_lead_rel_sigma = 0.25
            if contact_source.startswith("measured") and fol["kin"]["t"][-1] < t_contact - 0.2:
                follow_hidden_accel = _follow_accel_from_impact(fol["kin"], lead["kin"], L_f, t_contact)
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
                             "hidden_vehicle": hidden, "follower_hidden_accel_from_impact_mps2": follow_hidden_accel,
                             "road_m_per_image_row_at_follower": res_m_per_row,
                             "tracks_merged": merged, "follower_id": fol["track"]["id"], "lead_id": lead["track"]["id"],
                             "method": "constant-acceleration extrapolation of both vehicles from their last reliable "
                                       "positions to zero bumper gap (follower length prior "
                                       f"{FOLLOW_LENGTH_PRIOR[0]} +- {FOLLOW_LENGTH_PRIOR[1]} m)"}
        case["hazard_onset"] = hazard
        if t_contact is None and contact_source.startswith("not established"):
            stage("events", "UNDETERMINED", contact_source)
            case["status"] = "CONTACT_UNDETERMINED"
            case["speeds"] = _speed_summaries(vehicles, geo, prior, fps, rng, n_mc)
        elif t_contact is None:
            stage("events", "NO_CONTACT", "same-lane pair found; " + contact_source)
            case["status"] = "RECONSTRUCTED_NO_COLLISION"
            case["speeds"] = _speed_summaries(vehicles, geo, prior, fps, rng, n_mc)
        else:
            stage("events", "CONTACT", f"contact at t={t_contact:.2f}s; hazard onset ({hazard['source']}) t={hazard['t_s']:.2f}s")
            samples, speed_at_onset = _mc_samples(fol, lead, geo, prior, fps, hazard, rng, n_mc, a_lead,
                                                  a_lead_rel_sigma, t_contact if kink and kink["significant"] else None)
            if len(samples) >= 10:
                cf = assess_counterfactual(samples, hazard_info=hazard)
                case["counterfactual"] = asdict(cf)
            else:
                case["counterfactual_error"] = (f"only {len(samples)} of {n_mc} Monte Carlo reconstructions were valid "
                                                "(too few measured positions under resampled geometry)")
            case["speeds"] = _speed_summaries(vehicles, geo, prior, fps, rng, n_mc, ref_t=hazard["t_s"])
            case["follower_speed_at_hazard_onset"] = speed_at_onset
            impacts = [s.t_impact_s for s in samples if s.t_impact_s is not None]
            case["collision"]["t_contact_ci90_s"] = ([float(np.percentile(impacts, 5)), float(np.percentile(impacts, 95))]
                                                      if impacts else None)
            case["status"] = "RECONSTRUCTED_COLLISION"
    if pair is not None and len(pair) >= 2:
        # surrogate safety measures; own random stream (spawned) so every other result stays bit-identical
        col = case.get("collision") or {}
        t_cut_nm = col.get("t_contact_s")
        res_nm = col.get("road_m_per_image_row_at_follower")
        if case.get("status") == "CONTACT_UNDETERMINED":
            # a smallest-gap figure would contradict "the footage cannot exclude contact"
            case["near_miss"] = {"withheld": "whether the vehicles touched is not established, so a closest-approach "
                                             "figure would be misleading"}
        elif res_nm is not None and res_nm > 2 * FOLLOW_LENGTH_PRIOR[1]:
            case["near_miss"] = {"withheld": f"one image row spans {res_nm:.2f} m of road at the follower: the bumper gap "
                                             f"is not resolved finer than the vehicle-length uncertainty"}
        else:
            nm = _near_miss(pair[0], pair[1], geo, prior, fps, rng.spawn(1)[0], max(40, n_mc // 2),
                            pair[0].get("direction", 1), t_max=(t_cut_nm - 0.3) if t_cut_nm else None)
            if nm:
                case["near_miss"] = nm
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
            kins_pre.append({**{kk: (vv[m] if isinstance(vv, np.ndarray) and len(vv) == len(m) else vv)
                                for kk, vv in k.items()}, "vehicle_id": v["track"]["id"]})
    subjects = {pair[0]["track"]["id"], pair[1]["track"]["id"]} if pair else None
    skep = cross_examine(frames, geo, prior, [v["boxes"] for v in vehicles], kins_pre,
                         probe[[0, -1]], enabled=skeptic_checks, subjects=subjects)
    withheld = set()
    for c in skep["checks"]:
        withheld |= set(c.get("withheld_vehicles") or [])
    if withheld and case.get("speeds"):
        for sp in case["speeds"]:
            if sp.get("vehicle_id") in withheld:
                sp["withheld"] = True
                sp["withheld_reason"] = "physically impossible reconstructed motion (tracking failure on this vehicle)"
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
        if isinstance(case.get("near_miss"), dict) and "min_gap_m" in case["near_miss"]:
            case["near_miss"]["disputed"] = True     # computed, kept for review, not reported as a finding
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
        elif case.get("counterfactual_error"):
            case["verdict"] = {"value": "INSUFFICIENT_EVIDENCE", "reason": case["counterfactual_error"]}
        elif case["status"] == "RECONSTRUCTED_NO_INTERACTION":
            # not a "no collision" finding: the tracker may have fused or lost vehicles
            case["verdict"] = {"value": "NO_PAIR_RECONSTRUCTED",
                               "reason": "no two tracked vehicles were in the same lane and closing on each other, so no "
                                         "collision analysis was possible; the speeds of the tracked vehicles are reported"}
        elif case["status"] == "CONTACT_UNDETERMINED":
            case["verdict"] = {"value": "INSUFFICIENT_EVIDENCE", "reason": case["collision"]["source"]}
        else:
            case["verdict"] = {"value": "NO_COLLISION_ANALYSED",
                               "reason": "no contact between tracked vehicles was found in the footage"}

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


TTC_CAP_S = 10.0        # time-to-collision above this (or not closing) is reported as ">= 10 s": no conflict


def _near_miss(fol, lead, geo, prior, fps, rng, n_mc, direction, t_max=None, min_closing=0.3):
    """Surrogate safety measures over the interval in which BOTH vehicles were measured (nothing
    extrapolated): minimum bumper gap, minimum time-to-collision (gap / closing speed while closing,
    censored at TTC_CAP_S) and maximum deceleration rate to avoid the crash (DRAC = closing^2 / (2 gap)).
    Standard traffic-conflict indicators (Hayward 1972; Cooper & Ferguson 1976). Monte Carlo over the
    homography bootstrap, the marking prior, the follower-length prior and the smoother's own position /
    velocity uncertainty (one correlated offset per vehicle per draw)."""
    Hs = geo.H_samples or [geo.H]
    tf = {int(round(t * fps)) for t in _measured(fol["track"], fps)[0]}
    tl = {int(round(t * fps)) for t in _measured(lead["track"], fps)[0]}
    common = np.array(sorted(tf & tl), float) / fps
    if t_max is not None:
        common = common[common < t_max]
    if len(common) < 8:
        return None
    draws = {"min_gap_m": [], "min_ttc_s": [], "max_drac_mps2": []}
    t_ttc = []
    for _ in range(n_mc):
        Hm = Hs[rng.integers(len(Hs))]
        sx = 1.0 + rng.normal(0, prior.cycle_rel_sigma)
        kf = _kin_for(fol["track"], fps, Hm, sx=sx)
        kl = _kin_for(lead["track"], fps, Hm, sx=sx)
        if kf is None or kl is None:
            continue
        kf, kl = _aligned(kf, direction), _aligned(kl, direction)
        L = max(3.5, rng.normal(*FOLLOW_LENGTH_PRIOR))
        ef, el = rng.normal(0, 1, 2)
        gf, gl = rng.normal(0, 1, 2)
        xf = np.interp(common, kf["t"], kf["x"] + ef * kf["sx"])
        xl = np.interp(common, kl["t"], kl["x"] + el * kl["sx"])
        vf = np.interp(common, kf["t"], kf["v"] + gf * kf["sv"])
        vl = np.interp(common, kl["t"], kl["v"] + gl * kl["sv"])
        gap, clo = xl - xf - L, vf - vl
        draws["min_gap_m"].append(float(gap.min()))
        m = (clo > min_closing) & (gap > 0)
        ttc = gap[m] / clo[m] if m.any() else np.array([np.inf])
        k = int(np.argmin(ttc))
        draws["min_ttc_s"].append(float(min(ttc[k], TTC_CAP_S)))
        if m.any() and ttc[k] < TTC_CAP_S:
            t_ttc.append(float(common[m][k]))
        draws["max_drac_mps2"].append(float(np.max(clo[m] ** 2 / (2 * gap[m]))) if m.any() else 0.0)
    if len(draws["min_gap_m"]) < 10:
        return None
    out = {"observed_interval_s": [float(common[0]), float(common[-1])], "n_samples": len(draws["min_gap_m"]),
           "ttc_cap_s": TTC_CAP_S,
           "p_conflict_ttc_below_1_5s": float(np.mean(np.array(draws["min_ttc_s"]) < 1.5)),
           "t_min_ttc_s": float(np.median(t_ttc)) if len(t_ttc) >= 0.5 * len(draws["min_ttc_s"]) else None,
           "definitions": {"gap": "lead rear contact - follower rear contact - follower length (prior 4.6 +- 0.35 m)",
                           "ttc": f"gap / closing speed while closing faster than {min_closing} m/s; values above "
                                  f"{TTC_CAP_S:g} s (or never closing) are reported as {TTC_CAP_S:g} s",
                           "drac": "closing speed^2 / (2 gap): constant deceleration the follower needed to avoid contact"}}
    for k, v in draws.items():
        out[k] = {"median": float(np.median(v)), "ci90": [float(np.percentile(v, 5)), float(np.percentile(v, 95))]}
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
        a_f = None
        if t_contact_measured is not None:
            tc = t_contact_measured + rng.normal(0, 1.0 / fps)
            if kl["t"][-1] < tc - 0.2:
                dh = _lead_decel_from_impact(kf, kl, L_f, tc)
                if dh is not None:
                    a_l = -dh
            if kf["t"][-1] < tc - 0.2:
                a_f = _follow_accel_from_impact(kf, kl, L_f, tc)
        else:
            tc, _ = _extrapolate_contact(kf, kl, L_f, a_l)
        dv = rng.normal(0, float(np.interp(th, kf["t"], kf["sv"])))
        t = np.union1d(kf["t"], kl["t"])
        if tc is not None and tc > t[-1]:
            t = np.append(t, tc)
        x_l = _lead_x(kl, a_l, t)
        v_l = np.interp(t, kl["t"], kl["v"])
        beyond = t > kl["t"][-1]
        v_l[beyond] = np.maximum(0.0, kl["v"][-1] + a_l * (t[beyond] - kl["t"][-1]))
        a_end = a_l
        xf_t, vf_t = _follow_state(kf, t, a_f)
        kf = {**kf, "x": xf_t, "v": vf_t, "t": t}
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
