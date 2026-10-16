"""
The Skeptic, v2: cross-examination of an accepted reconstruction using
evidence the Reconstructor did not fit.

Why v1 (backend/agent/skeptic.py) was not enough. Its two checks re-used
the Reconstructor's own data: "depth trend" fits a line through the SAME
lane-width residuals the acceptance test already summarised, and "angle
ambiguity" re-reads the roll-search trace. Measured on the original demo it
also disputed a correct reconstruction (3.6 % speed error), and its single
validation was one reverted bug on one scene.

Each v2 check names the failure it can catch and the evidence it uses.
Each returns {check, status: PASS | DISPUTE | ADVISORY | NOT_APPLICABLE, statistic,
threshold, evidence, catches}.

  held_out_markings   leave-one-dash-out prediction of each dash centre
                      (pixels along the line). Catches mis-indexed or
                      irregular markings. Same feature family as the fit,
                      but out-of-sample.
  temporal_stability  geometry re-estimated independently on the first and
                      second halves of the clip; the two must map the same
                      pixels to the same road distances. Catches a camera
                      that moved, zoomed or was bumped (a single geometry
                      would then be wrong for part of the clip).
  marking_standard    measured dash duty cycle (dash length / cycle) vs the
                      assumed standard's. Catches a wrong marking standard
                      when the standards differ in shape (e.g. 3 m/9 m vs
                      6 m/12 m). CANNOT catch two standards with the same
                      duty cycle but different lengths (US 10/30 ft vs
                      Caltrans 12/36 ft) -- stated, not hidden.
  vehicle_width       each tracked vehicle's road-plane footprint width
                      (bottom edge of its box through the homography) must
                      be physically plausible and must not drift with
                      depth. Uses the vehicles, which the geometry never
                      saw. Tests the lateral scale and perspective, NOT the
                      longitudinal (speed) scale.
  kinematics          reconstructed pre-impact speeds and accelerations
                      must be physically possible (lower 2-sigma bound of
                      |a| averaged over 0.4 s <= 12 m/s^2, of v <= 70 m/s),
                      per vehicle.
                      Catches gross scale or tracking errors. Blocks only
                      when a vehicle the verdict rests on fails; a failing
                      bystander has its speeds withheld (ADVISORY).

What no check here can establish: the true length of the marking cycle. On
a static camera looking along the road it fixes the speed scale and is not
identifiable from the image (see docs/technical_report.md, "What the video
cannot tell us"). It enters as a stated prior with its own uncertainty.
"""
from __future__ import annotations

import numpy as np

from backend.geometry.road_geometry import (
    RoadGeometry, RoadPrior, estimate_road_geometry, image_to_road, median_background,
)

THRESH = {
    "held_out_rms_px": 1.5,
    "held_out_max_px": 4.0,
    "temporal_scale_rel": 0.02,
    "temporal_vp_px": 4.0,
    "duty_abs": 0.06,
    "width_range_m": (1.3, 3.0),
    "width_drift_rel": 0.5,         # 3D bodies' footprint edge drifts up to ~35 % with depth even when geometry is right (dev)
    "max_decel_mps2": 12.0,          # ~1.2 g: beyond tyre-road limits even after a 2-sigma allowance
    "max_speed_mps": 70.0,
}


def _result(check, status, statistic, threshold, evidence, catches):
    return {"check": check, "status": status, "statistic": statistic, "threshold": threshold,
            "evidence": evidence, "catches": catches}


def check_held_out_markings(geo: RoadGeometry):
    catches = "mis-indexed or irregular dashes"
    d = geo.dashes
    if len(d) < 4:
        return _result("held_out_markings", "NOT_APPLICABLE", None, None, "fewer than 4 dashes", catches)
    P = geo.prior["dash_len_m"] + geo.prior["dash_gap_m"]
    L = geo.prior["dash_len_m"]
    cen = np.array([(x["s_near"] + x["s_far"]) / 2.0 for x in d])
    X = np.array([x["k"] * P + L / 2.0 for x in d])
    errs = []
    for i in range(len(d)):
        m = np.ones(len(d), bool)
        m[i] = False
        w = np.sqrt(1.0 / cen[m] ** 4)
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), X[m]]) * w[:, None], cen[m] * w, rcond=None)
        pred = coef[0] + coef[1] * X[i]
        errs.append(abs(pred - cen[i]) / cen[i] ** 2)
    errs = np.array(errs)
    # the two end dashes are EXTRAPOLATED by their leave-one-out fit; judge the worst case on interior dashes
    interior = errs[1:-1] if len(errs) > 3 else errs
    rms, mx = float(np.sqrt(np.mean(errs ** 2))), float(interior.max())
    bad = rms > THRESH["held_out_rms_px"] or mx > THRESH["held_out_max_px"]
    return _result("held_out_markings", "DISPUTE" if bad else "PASS",
                   {"rms_px": rms, "max_px": mx, "worst_dash": int(np.argmax(errs))},
                   {"rms_px": THRESH["held_out_rms_px"], "max_px": THRESH["held_out_max_px"]},
                   f"each dash centre predicted from the others: RMS {rms:.2f}px, worst {mx:.2f}px", catches)


_HALF_CACHE = {}


def _half_geometries(frames, prior, half):
    """The two half-clip geometries depend only on the frames and the prior, not on the tracks, so a
    re-run of the reasoning on the same clip (the investigator's remedies) reuses them."""
    f0 = frames[0]
    key = (id(frames), len(frames), f0.shape, int(f0[::37, ::41].sum()), repr(prior))
    if key not in _HALF_CACHE:
        if len(_HALF_CACHE) > 8:
            _HALF_CACHE.clear()
        _HALF_CACHE[key] = (estimate_road_geometry(frames[:half], prior), estimate_road_geometry(frames[half:], prior))
    return _HALF_CACHE[key]


def check_temporal_stability(frames, geo: RoadGeometry, prior: RoadPrior, probe_uv):
    catches = "camera moved / zoomed / bumped during the clip"
    if len(frames) < 40 or geo.H is None:
        return _result("temporal_stability", "NOT_APPLICABLE", None, None, "clip too short to split", catches)
    half = len(frames) // 2
    g1, g2 = _half_geometries(frames, prior, half)
    if not (g1.ok and g2.ok):
        which = [n for n, g in (("first", g1), ("second", g2)) if not g.ok]
        # failing to re-fit a half-clip (e.g. vehicles hiding an edge line) is not evidence of camera motion
        return _result("temporal_stability", "NOT_APPLICABLE", {"failed_halves": which}, None,
                       f"geometry could not be re-established on the {' and '.join(which)} half of the clip "
                       f"({'; '.join((g1.reasons if not g1.ok else []) + (g2.reasons if not g2.ok else []))})", catches)
    uv = np.asarray(probe_uv, float)
    p1, p2 = image_to_road(g1.H, uv), image_to_road(g2.H, uv)
    span1 = np.linalg.norm(p1[-1] - p1[0])
    span2 = np.linalg.norm(p2[-1] - p2[0])
    rel = float(abs(span1 / span2 - 1.0)) if span2 > 0 else np.inf
    dvp = float(np.linalg.norm(np.array(g1.vp) - np.array(g2.vp)))
    bad = rel > THRESH["temporal_scale_rel"] or dvp > THRESH["temporal_vp_px"]
    return _result("temporal_stability", "DISPUTE" if bad else "PASS",
                   {"scale_rel_diff": rel, "vp_shift_px": dvp},
                   {"scale_rel_diff": THRESH["temporal_scale_rel"], "vp_shift_px": THRESH["temporal_vp_px"]},
                   f"first vs second half of the clip: road-distance scale differs by {rel:.2%}, "
                   f"vanishing point moved {dvp:.1f}px", catches)


def check_marking_standard(geo: RoadGeometry):
    catches = "wrong marking standard (different dash/gap proportions)"
    dm, dp = geo.diagnostics.get("duty_measured"), geo.diagnostics.get("duty_prior")
    if dm is None or dp is None:
        return _result("marking_standard", "NOT_APPLICABLE", None, None, "duty cycle unavailable", catches)
    diff = float(abs(dm - dp))
    return _result("marking_standard", "DISPUTE" if diff > THRESH["duty_abs"] else "PASS",
                   {"duty_measured": dm, "duty_prior": dp}, {"abs_diff": THRESH["duty_abs"]},
                   f"dashes cover {dm:.0%} of each cycle in the image; the assumed standard says {dp:.0%}", catches)


def check_vehicle_width(geo: RoadGeometry, tracks_spans):
    """tracks_spans: per track, list of ((u_left, u_right), v_bottom) -- the
    horizontal extent of the vehicle's lowest pixel rows (its footprint edge
    nearest the camera) at measured frames."""
    catches = "wrong lateral scale / perspective (lane-width prior, horizon)"
    per = []
    for spans in tracks_spans:
        if len(spans) < 8:
            continue
        left = np.array([[sp[0][0], sp[1]] for sp in spans])
        right = np.array([[sp[0][1], sp[1]] for sp in spans])
        pl, pr = image_to_road(geo.H, left), image_to_road(geo.H, right)
        wv = np.abs(pr[:, 1] - pl[:, 1])
        xm = 0.5 * (pl[:, 0] + pr[:, 0])
        if np.ptp(xm) < 10:
            continue
        slope = np.polyfit(xm, wv, 1)[0]
        drift = float(abs(slope * np.ptp(xm)) / np.median(wv))
        per.append({"median_width_m": float(np.median(wv)), "drift_rel": drift, "x_span_m": float(np.ptp(xm))})
    if not per:
        return _result("vehicle_width", "NOT_APPLICABLE", None, None, "no vehicle tracked over >= 10 m", catches)
    lo, hi = THRESH["width_range_m"]
    bad_w = [p for p in per if not (lo <= p["median_width_m"] <= hi)]
    bad_d = [p for p in per if p["drift_rel"] > THRESH["width_drift_rel"]]
    status = "DISPUTE" if (bad_w or bad_d) else "PASS"
    ev = "; ".join(f"vehicle {i}: {p['median_width_m']:.2f} m wide, drift {p['drift_rel']:.0%} over {p['x_span_m']:.0f} m"
                   for i, p in enumerate(per))
    return _result("vehicle_width", status, {"vehicles": per},
                   {"width_range_m": list(THRESH["width_range_m"]), "drift_rel": THRESH["width_drift_rel"]}, ev, catches)


ACCEL_WINDOW_S = 0.4


def _windowed_accel_lower_bound(t, v, sv, win=ACCEL_WINDOW_S):
    if len(t) < 3 or t[-1] - t[0] < win:
        return float(np.max(np.abs(np.diff(v) / np.maximum(np.diff(t), 1e-6)))) if len(t) > 1 else 0.0
    j = np.searchsorted(t, t + win)
    ok = j < len(t)
    i, j = np.where(ok)[0], j[ok]
    dt = t[j] - t[i]
    a = (v[j] - v[i]) / dt
    sa = np.sqrt(sv[i] ** 2 + sv[j] ** 2) / dt
    return float(np.max(np.abs(a) - 2 * sa))


def check_kinematics(kins, subjects=None):
    """kins: per vehicle, smoothed kinematics (dict with t, v, a, sv, sa and
    optionally 'vehicle_id'). subjects: ids of the vehicles the verdict rests
    on (follower and lead), or None when there is no interacting pair.

    Per-vehicle test. A physically impossible trajectory of a BYSTANDER
    vehicle is a tracking failure on that vehicle: its speeds are withheld
    (status ADVISORY), but it does not block a verdict about two other
    vehicles. A failure of a subject vehicle blocks (DISPUTE). With no pair,
    the check disputes only if most trajectories fail (a global scale error
    moves all of them; the geometry checks cover that case independently)."""
    catches = "gross scale or tracking errors"
    if not kins:
        return _result("kinematics", "NOT_APPLICABLE", None, None, "no trajectories", catches)
    per = []
    for n, k in enumerate(kins):
        # acceleration AVERAGED over ACCEL_WINDOW_S (speed change / time), lower 2-sigma bound: a smoother
        # overshoots for a frame or two at a sharp brake onset or at the end of a track, which is not
        # evidence of impossible motion; a wrong scale or a tracking jump moves the average too
        a = _windowed_accel_lower_bound(np.asarray(k["t"]), np.asarray(k["v"]), np.asarray(k["sv"]))
        v = float(np.max(np.abs(k["v"]) - 2 * k["sv"]))
        per.append({"vehicle_id": k.get("vehicle_id", n), "max_abs_accel_mps2": a, "max_speed_mps": v,
                    "plausible": not (a > THRESH["max_decel_mps2"] or v > THRESH["max_speed_mps"])})
    bad = [p for p in per if not p["plausible"]]
    if subjects is not None:
        bad_subj = [p for p in bad if p["vehicle_id"] in subjects]
        status = "DISPUTE" if bad_subj else ("ADVISORY" if bad else "PASS")
    else:
        status = "DISPUTE" if len(bad) > len(per) / 2 else ("ADVISORY" if bad else "PASS")
    worst_a = max(p["max_abs_accel_mps2"] for p in per)
    worst_v = max(p["max_speed_mps"] for p in per)
    ev = "; ".join(f"vehicle {p['vehicle_id']}{' (subject)' if subjects and p['vehicle_id'] in subjects else ''}: "
                   f"|a| {p['max_abs_accel_mps2']:.1f} m/s^2, v {p['max_speed_mps']:.1f} m/s"
                   f"{'' if p['plausible'] else ' -- implausible'}" for p in per)
    res = _result("kinematics", status, {"max_abs_accel_mps2": worst_a, "max_speed_mps": worst_v, "vehicles": per},
                  {"max_abs_accel_mps2": THRESH["max_decel_mps2"], "max_speed_mps": THRESH["max_speed_mps"]},
                  ev + " (pre-impact, lower 2-sigma bounds)", catches)
    res["withheld_vehicles"] = [p["vehicle_id"] for p in bad]
    return res


def cross_examine(frames, geo: RoadGeometry, prior: RoadPrior, tracks_boxes, kins, probe_uv, enabled=None,
                  subjects=None):
    enabled = enabled or ["held_out_markings", "temporal_stability", "marking_standard", "vehicle_width", "kinematics"]
    out = []
    if "held_out_markings" in enabled:
        out.append(check_held_out_markings(geo))
    if "temporal_stability" in enabled:
        out.append(check_temporal_stability(frames, geo, prior, probe_uv))
    if "marking_standard" in enabled:
        out.append(check_marking_standard(geo))
    if "vehicle_width" in enabled:
        out.append(check_vehicle_width(geo, tracks_boxes))
    if "kinematics" in enabled:
        out.append(check_kinematics(kins, subjects))
    disputed = [c["check"] for c in out if c["status"] == "DISPUTE"]
    return {"disputed": bool(disputed), "disputed_checks": disputed, "checks": out,
            "untestable": ["marking cycle length (longitudinal / speed scale): a stated prior, not verifiable "
                           "from a single static view"]}
