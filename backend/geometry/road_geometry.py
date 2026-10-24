"""
Static-camera road geometry, v2 (replaces the single-frame, row-based,
integer-roll-search geometry in backend/agent/loop.py + cv/geometry/roll_correction.py,
which are kept only for the ablation in evaluation/experiments).

Input: frames from a fixed camera that sees a straight road with two solid
edge lines and one dashed line midway between them.

1. Geometry frame = per-pixel temporal median of frames sampled across the
   clip: moving vehicles vanish, sensor noise drops by ~sqrt(N).
2. Road vanishing point (RANSAC over Hough segments) and the three lines
   (left edge, dashed centre, right edge), classified solid/dashed from
   their own intensity profile.
3. Horizon (vanishing line of the road plane) = harmonic conjugate of the
   centre line w.r.t. the edges -> camera roll in closed form.
4. Dashes measured along the centre line in s = 1/d (affine in road
   distance), indexed by a robust period fit that tolerates missing or
   occluded dashes.
5. Correspondences on transversals parallel to the horizon at every fitted
   dash end (left edge, centre, right edge) -> image->road homography.
   Approximation, stated: lines of constant road distance are taken
   parallel to the horizon in the image, which is exact for zero camera
   yaw; the error grows like W*tan(yaw) for points off the centre line.
6. Uncertainty: bootstrap over dashes and over the line-angle estimates ->
   a set of homography samples, carried downstream (positions, speeds,
   avoidability are all evaluated on every sample).

Every metric number this module produces depends on two stated priors:
the dash cycle length (longitudinal scale -- speed) and the edge-to-centre
distance (lateral scale only). They are inputs, never constants buried in
code, and their uncertainty is part of the reported interval.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import cv2
import numpy as np

from cv.geometry.projective import (
    min_distance_inside, line_segments, vanishing_point_ransac, line_directions_about_vp, direction_vector,
    harmonic_horizon_direction, intersect_lines, strip_along_line, max_distance_inside, sample_strip,
)


@dataclass
class RoadPrior:
    """Physical marking dimensions assumed for this road. Defaults are the
    Stradiva synthetic standard; real deployments must set them per
    jurisdiction/road class (see docs/assumptions.md)."""
    name: str = "carai_synthetic_centerline"
    edge_to_centre_m: float = 1.85
    dash_len_m: float = 3.0
    dash_gap_m: float = 9.0
    cycle_rel_sigma: float = 0.03          # 1-sigma relative uncertainty of dash cycle (longitudinal scale)
    width_rel_sigma: float = 0.05          # 1-sigma relative uncertainty of edge-to-centre distance (lateral scale)

    @property
    def cycle_m(self):
        return self.dash_len_m + self.dash_gap_m


@dataclass
class GeometryConfig:
    n_median_frames: int = 48
    n_boot: int = 200
    min_dashes: int = 3
    min_period_snr: float = 4.0
    max_dash_resid_px: float = 2.5         # RMS residual of dash centres along the line, pixels
    seed: int = 0


@dataclass
class RoadGeometry:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    H: Optional[np.ndarray] = None                 # image (u,v) -> road (X,Y) metres
    H_samples: List[np.ndarray] = field(default_factory=list)
    vp: Optional[List[float]] = None
    horizon_dir: Optional[List[float]] = None
    roll_deg: Optional[float] = None
    lines: dict = field(default_factory=dict)
    dashes: List[dict] = field(default_factory=list)
    s_affine: Optional[List[float]] = None         # s = a + b * X  (X in metres along road)
    calibrated_x_range_m: Optional[List[float]] = None
    diagnostics: dict = field(default_factory=dict)
    prior: dict = field(default_factory=dict)

    def to_json(self):
        d = asdict(self)
        d["H"] = None if self.H is None else np.asarray(self.H).tolist()
        d["H_samples"] = f"{len(self.H_samples)} samples (omitted)"
        return d


# ------------------------------------------------------------------ helpers --

def median_background(frames, n=48, portion=(0.0, 1.0)):
    lo = int(portion[0] * (len(frames) - 1))
    hi = int(portion[1] * (len(frames) - 1))
    idx = np.linspace(lo, hi, min(n, hi - lo + 1)).astype(int)
    stack = np.stack([frames[i] for i in idx])
    return np.median(stack, axis=0).astype(np.uint8)


def event_background(frames, n=48, min_frames=30):
    """Background for event footage: temporal median of the FIRST HALF of the
    clip. Vehicles that come to rest after a collision (later in the clip)
    would otherwise enter a whole-clip median and leave a 'ghost' in front of
    the moving vehicles. Also returns where the first- and second-half
    backgrounds disagree (objects that came to rest or left), as evidence."""
    if len(frames) < 2 * min_frames:
        return median_background(frames, n), None
    b1 = median_background(frames, n, (0.0, 0.5))
    b2 = median_background(frames, n, (0.5, 1.0))
    diff = cv2.absdiff(b1, b2).max(axis=2) > 30
    diff = cv2.morphologyEx(diff.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(diff)
    regions = [{"bbox": [int(v) for v in stats[i][:4]], "area_px": int(stats[i][4])}
               for i in range(1, n_lbl) if stats[i][4] > 150]
    return b1, regions


def _profile_runs(prof, s, asphalt_level=None):
    """Binary paint/asphalt segmentation of a 1D profile with sub-sample
    edges. asphalt_level comes from strips beside the line (a solid line's
    own profile contains no asphalt to compare against). Returns
    (runs, asphalt, paint) with runs = [(s_lo, s_hi)] not touching the ends."""
    lo = np.percentile(prof, 30) if asphalt_level is None else asphalt_level
    hi = np.percentile(prof, 97)
    if hi - lo < 15:
        return [], lo, hi
    thr = 0.5 * (lo + hi)
    above = prof > thr
    runs = []
    i, n = 0, len(prof)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            if i > 0 and j < n - 1:
                fi = (i - 1) + (thr - prof[i - 1]) / (prof[i] - prof[i - 1] + 1e-9)
                fj = j + (prof[j] - thr) / (prof[j] - prof[j + 1] + 1e-9)
                si = np.interp(fi, np.arange(n), s)
                sj = np.interp(fj, np.arange(n), s)
                runs.append((min(si, sj), max(si, sj)))
            i = j + 1
        else:
            i += 1
    return runs, lo, hi


def _classify_line(gray, vp, angle, w, h):
    dmax = max_distance_inside(vp, angle, w, h)
    if dmax is None or dmax < 60:
        return None
    dmin = max(15.0, dmax / 25.0, min_distance_inside(vp, angle, w, h))
    if dmax < 2 * dmin:
        return None
    maps = strip_along_line(vp, angle, dmin, dmax, n_samples=1024)
    prof = sample_strip(gray, maps)
    side = []
    for off in (-2.0, 2.0):
        dm = max_distance_inside(vp, angle + off, w, h)
        if dm is not None and dm > dmin * 2:
            side.append(np.median(sample_strip(gray, strip_along_line(vp, angle + off, dmin, min(dm, dmax), n_samples=256), reduce="mean")))
    asphalt = float(np.median(side)) if side else None
    runs, lo, hi = _profile_runs(prof, maps[2], asphalt)
    s = maps[2]
    span = s[0] - s[-1]
    thr = 0.5 * (lo + hi)
    paint_frac = float(np.mean(prof > thr)) if hi - lo >= 15 else 0.0
    return {"angle_deg": float(angle), "d_range": [float(dmin), float(dmax)], "paint_frac": paint_frac,
            "n_runs": len(runs), "contrast": float(hi - lo), "runs": runs}


def _fit_dash_index(dash_runs, prior: RoadPrior, cfg: GeometryConfig):
    """dash_runs: list of (s_near, s_far) with s_near < s_far (nearer = smaller s).

    Uses dash CENTRES: blur/anti-aliasing shifts both ends of a dash outward
    by a roughly constant number of pixels, which in s = 1/d is a bias that
    grows like 1/d^2 -- it cancels in the centre to first order. Indices are
    assigned from a robust base spacing (tolerates missing/occluded dashes),
    then s_centre = a + b*X is fitted by weighted least squares with weights
    1/sigma_s^2, sigma_s = sigma_px * s^2 (a pixel of noise is a much larger
    error in s far away). Residual reported in PIXELS along the line."""
    if len(dash_runs) < cfg.min_dashes:
        return None
    runs = sorted(dash_runs)
    lens = np.array([r[1] - r[0] for r in runs])
    centres = np.array([(r[0] + r[1]) / 2.0 for r in runs])
    # expected s-length grows mildly with distance because of blur; compare to neighbours, drop merged/partial runs
    ok = np.ones(len(runs), bool)
    for i in range(len(runs)):
        nb = lens[max(0, i - 2): i + 3]
        ok[i] = 0.5 * np.median(nb) <= lens[i] <= 1.6 * np.median(nb)
    runs = [r for r, o in zip(runs, ok) if o]
    centres = centres[ok]
    if len(runs) < cfg.min_dashes:
        return None
    gaps = np.diff(centres)
    g_sorted = np.sort(gaps)
    base = np.median(g_sorted[: max(1, len(g_sorted) // 2 + 1)])
    # refine the base by fitting increments to near-integer multiples
    mult = np.maximum(1, np.round(gaps / base))
    base = float(np.sum(gaps) / np.sum(mult))
    k = np.concatenate([[0], np.cumsum(np.round(gaps / base).astype(int))])
    if len(set(k.tolist())) != len(k):
        return None
    P, L = prior.cycle_m, prior.dash_len_m
    X = k * P + L / 2.0
    wts = 1.0 / np.maximum(centres, 1e-9) ** 4
    A = np.column_stack([np.ones_like(X), X])
    Wm = np.sqrt(wts)[:, None]
    coef, *_ = np.linalg.lstsq(A * Wm, centres * Wm.ravel(), rcond=None)
    resid_s = centres - A @ coef
    resid_px = resid_s / np.maximum(centres, 1e-9) ** 2      # ds = -dd/d^2  ->  dd = ds * d^2
    period_s = coef[1] * P
    duty = float(np.median([(r[1] - r[0]) for r in runs[: max(2, len(runs) // 2)]]) / abs(period_s)) if period_s else None
    return {"a": float(coef[0]), "b": float(coef[1]), "k": k.tolist(), "runs": runs, "centres": centres.tolist(),
            "resid_px_rms": float(np.sqrt(np.mean(resid_px ** 2))), "resid_px_max": float(np.max(np.abs(resid_px))),
            "duty_measured": duty, "duty_prior": L / P,
            "X_range": [float(k.min() * P), float(k.max() * P + L)]}


def _duty_on_line_axis(gray, vp, angle_deg, d_range, fit, prior: RoadPrior):
    """Dash duty cycle (dash length / cycle) measured on the refined line's
    AXIS only. The detection strip is wide (it must tolerate an imprecise
    line) and takes the maximum across it; a dash end crosses that strip
    obliquely, so the wide strip lengthens every dash by a perspective-
    dependent amount and inflates the duty cycle, most for near dashes and
    yawed cameras. A 1-2 px wide sample along the sub-pixel axis does not.
    Nearer half of the dashes (least blur in s). None if < 2 usable dashes."""
    dmin, dmax = d_range
    maps = strip_along_line(vp, angle_deg, dmin, dmax, n_samples=2048, half_width_frac=0.002, n_perp=3, min_half_px=0.6)
    prof = sample_strip(gray, maps, reduce="mean")
    runs, lo, hi = _profile_runs(prof, maps[2])
    if not runs or not fit.get("b"):
        return None
    period_s = abs(fit["b"] * prior.cycle_m)
    lens = sorted(((r[0] + r[1]) / 2.0, r[1] - r[0]) for r in runs)
    expected = prior.dash_len_m / prior.cycle_m * period_s
    lens = [ln for c, ln in lens if 0.3 * expected < ln < 2.5 * expected]   # whole dashes only
    if len(lens) < 2:
        return None
    near = lens[: max(2, len(lens) // 2)]
    return float(np.median(near) / period_s)


def refine_line(gray, vp, angle_deg, d_min, d_max, n=90):
    """Sub-pixel stripe-centre points along a candidate line, then a robust
    (Huber) cv2.fitLine. Returns (point, unit_direction_pointing_down, rms_px, n_pts) or None."""
    dvec = direction_vector(angle_deg)
    nvec = np.array([dvec[1], -dvec[0]])
    pts = []
    for d in np.geomspace(d_min, d_max, n):
        half = max(5.0, 0.03 * d)
        offs = np.arange(-half, half + 0.5, 0.5)
        c = vp + d * dvec
        xy = c[None, :] + offs[:, None] * nvec[None, :]
        prof = cv2.remap(gray, xy[:, 0].astype(np.float32).reshape(1, -1), xy[:, 1].astype(np.float32).reshape(1, -1),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).ravel().astype(np.float32)
        base = np.percentile(prof, 25)
        peak = prof.max()
        if peak - base < 25:
            continue
        m = prof > base + 0.5 * (peak - base)
        idx = np.where(m)[0]
        if len(idx) == 0 or idx[0] == 0 or idx[-1] == len(prof) - 1:
            continue
        wv = prof[idx] - base
        off = float(np.sum(offs[idx] * wv) / np.sum(wv))
        pts.append(c + off * nvec)
    if len(pts) < 8:
        return None
    pts = np.array(pts, np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    dirv = np.array([vx, vy], float)
    if dirv[1] < 0:
        dirv = -dirv
    p0 = np.array([x0, y0], float)
    rel = pts - p0
    perp = rel @ np.array([dirv[1], -dirv[0]])
    return p0, dirv, float(np.sqrt(np.mean(perp ** 2))), len(pts)


def lines_intersection(lines):
    """Least-squares intersection of 2D lines given as (point, direction)."""
    A, b = [], []
    for p, d in lines:
        nrm = np.array([d[1], -d[0]])
        A.append(nrm)
        b.append(nrm @ p)
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return sol


def angle_of(direction):
    return float(np.degrees(np.arctan2(direction[0], direction[1])))


def _homography_from(vp, ang_l, ang_c, ang_r, hd, a, b, X_values, prior: RoadPrior):
    dl, dc, dr = direction_vector(ang_l), direction_vector(ang_c), direction_vector(ang_r)
    img, wld = [], []
    W = prior.edge_to_centre_m
    for X in X_values:
        s = a + b * X
        if s <= 0:
            continue
        c = vp + (1.0 / s) * dc
        pl = intersect_lines(c, hd, vp, dl)
        pr = intersect_lines(c, hd, vp, dr)
        if pl is None or pr is None:
            continue
        img += [pl, c, pr]
        wld += [(X, -W), (X, 0.0), (X, W)]
    if len(img) < 6:
        return None
    H, _ = cv2.findHomography(np.array(img, np.float32), np.array(wld, np.float32), 0)
    return H


def image_to_road(H, uv):
    pts = np.atleast_2d(np.asarray(uv, dtype=np.float32)).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def road_to_image(H, xy):
    pts = np.atleast_2d(np.asarray(xy, dtype=np.float32)).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(np.asarray(H, dtype=np.float64))).reshape(-1, 2)


# --------------------------------------------------------------------- main --

def estimate_road_geometry(frames, prior: RoadPrior = None, cfg: GeometryConfig = None, background=None):
    prior = prior or RoadPrior()
    cfg = cfg or GeometryConfig()
    rng = np.random.default_rng(cfg.seed)
    geo = RoadGeometry(ok=False, prior=asdict(prior))
    bg = background if background is not None else median_background(frames, cfg.n_median_frames)
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY) if bg.ndim == 3 else bg
    h, w = gray.shape

    segs = line_segments(gray, roi_top_frac=0.0)
    vp, inl = vanishing_point_ransac(segs, rng)
    if vp is None:
        geo.reasons.append("no road vanishing point: lane lines not detected")
        return geo
    geo.vp = vp.tolist()
    dirs = line_directions_about_vp(segs, inl, vp, max_lines=6)
    cls = [c for c in (_classify_line(gray, vp, d["angle_deg"], w, h) for d in dirs) if c is not None]
    for c in cls:
        if c["paint_frac"] >= 0.6 or (c["n_runs"] <= 2 and c["paint_frac"] >= 0.45):
            c["kind"] = "solid"
        elif c["n_runs"] >= cfg.min_dashes and 0.08 <= c["paint_frac"] < 0.6:
            c["kind"] = "dashed"
        else:
            c["kind"] = "other"
    geo.diagnostics["line_candidates"] = [{k: v for k, v in c.items() if k != "runs"} for c in cls]
    dashed = [c for c in cls if c["kind"] == "dashed"]
    if not dashed:
        geo.reasons.append("no dashed marking found: longitudinal metric scale is not identifiable from this view")
        return geo
    centre = max(dashed, key=lambda c: c["n_runs"])
    solids = [c for c in cls if c["kind"] == "solid"]
    # direction_vector(angle) = (sin a, cos a): a smaller angle lies to the image LEFT going down from the VP
    left = [c for c in solids if c["angle_deg"] < centre["angle_deg"]]
    right = [c for c in solids if c["angle_deg"] > centre["angle_deg"]]
    if not left or not right:
        geo.reasons.append("a road edge line is missing on one side: horizon/roll and lateral scale not identifiable")
        return geo
    le = min(left, key=lambda c: centre["angle_deg"] - c["angle_deg"])
    re = min(right, key=lambda c: c["angle_deg"] - centre["angle_deg"])
    geo.lines = {"left_edge_deg": le["angle_deg"], "centre_deg": centre["angle_deg"], "right_edge_deg": re["angle_deg"]}

    # sub-pixel refinement of the three lines, then their least-squares common point
    refined = {}
    for key, c in (("left", le), ("centre", centre), ("right", re)):
        r = refine_line(gray, vp, c["angle_deg"], c["d_range"][0], c["d_range"][1])
        if r is None:
            geo.reasons.append(f"could not refine the {key} line to sub-pixel precision")
            return geo
        refined[key] = r
    vp = lines_intersection([(r[0], r[1]) for r in refined.values()])
    geo.vp = vp.tolist()
    angs = {k: angle_of(r[1]) for k, r in refined.items()}
    geo.lines = {"left_edge_deg": angs["left"], "centre_deg": angs["centre"], "right_edge_deg": angs["right"],
                 "fit_rms_px": {k: r[2] for k, r in refined.items()}}
    le = {**le, "angle_deg": angs["left"]}
    re = {**re, "angle_deg": angs["right"]}
    dmax_c = max_distance_inside(vp, angs["centre"], w, h)
    centre = _classify_line(gray, vp, angs["centre"], w, h) or centre

    hd = harmonic_horizon_direction(vp, le["angle_deg"], centre["angle_deg"], re["angle_deg"])
    geo.horizon_dir = hd.tolist()
    # sign: positive = camera rolled clockwise as seen by the camera (matches evaluation/sim/world.py);
    # verified against simulator ground truth in evaluation/experiments (roll sweep)
    geo.roll_deg = float(-np.degrees(np.arctan2(hd[1], hd[0])))

    # dashes along the centre line (s = 1/d; nearer = smaller s)
    runs = [(min(r), max(r)) for r in centre["runs"]]
    fit = _fit_dash_index(runs, prior, cfg)
    if fit is None:
        geo.reasons.append(f"fewer than {cfg.min_dashes} complete dashes could be indexed on the centre line")
        return geo
    geo.dashes = [{"s_near": r[0], "s_far": r[1], "k": int(k)} for r, k in zip(fit["runs"], fit["k"])]
    geo.s_affine = [fit["a"], fit["b"]]
    duty_narrow = _duty_on_line_axis(gray, vp, angs["centre"], centre["d_range"], fit, prior)
    geo.diagnostics.update({"dash_resid_px_rms": fit["resid_px_rms"], "dash_resid_px_max": fit["resid_px_max"],
                            "duty_measured": duty_narrow if duty_narrow is not None else fit["duty_measured"],
                            "duty_measured_wide_strip": fit["duty_measured"], "duty_prior": fit["duty_prior"],
                            "n_dashes": len(fit["runs"])})
    if fit["resid_px_rms"] > cfg.max_dash_resid_px:
        geo.reasons.append(f"dash centres are not equally spaced along the line (RMS residual {fit['resid_px_rms']:.1f}px): "
                           f"curved road, wrong vanishing point or irregular marking")
        return geo

    x_lo, x_hi = fit["X_range"]
    xs = np.arange(x_lo, x_hi + 1e-6, prior.cycle_m / 2.0)
    H = _homography_from(vp, le["angle_deg"], centre["angle_deg"], re["angle_deg"], hd, fit["a"], fit["b"], xs, prior)
    if H is None:
        geo.reasons.append("homography construction failed")
        return geo
    geo.H = H
    geo.calibrated_x_range_m = [float(x_lo), float(x_hi)]

    # ---- bootstrap: dashes + line-angle jitter (angle sigma from the segment scatter) ----
    # per-line angle sigma from the sub-pixel fit (rms / (length * sqrt(n/12))), floor 0.01 deg
    sig = {}
    for key, c in (("left", le), ("centre", centre), ("right", re)):
        p0, dv, rms, npts = refined[key]
        length = max(1.0, c["d_range"][1] - c["d_range"][0])
        sig[key] = max(0.01, float(np.degrees(max(rms, 0.2) / (length * np.sqrt(npts / 12.0)))))
    geo.diagnostics["line_angle_sigma_deg"] = sig
    samples = []
    runs_arr = fit["runs"]
    k_arr = np.array(fit["k"])
    P, L = prior.cycle_m, prior.dash_len_m
    for _ in range(cfg.n_boot):
        # resample dashes, keep their (already determined) cycle indices -> refit s = a + b X
        idx = rng.choice(len(runs_arr), len(runs_arr), replace=True)
        if len(set(idx.tolist())) < 2:
            continue
        cen = np.array(fit["centres"])
        X = k_arr[idx] * P + L / 2.0
        S = cen[idx]
        wts = np.sqrt(1.0 / np.maximum(S, 1e-9) ** 4)
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(X), X]) * wts[:, None], S * wts, rcond=None)
        jl, jc, jr = rng.normal(0, sig["left"]), rng.normal(0, sig["centre"]), rng.normal(0, sig["right"])
        hd2 = harmonic_horizon_direction(vp, le["angle_deg"] + jl, centre["angle_deg"] + jc, re["angle_deg"] + jr)
        H2 = _homography_from(vp, le["angle_deg"] + jl, centre["angle_deg"] + jc, re["angle_deg"] + jr, hd2,
                              coef[0], coef[1], xs, prior)
        if H2 is not None:
            samples.append(H2)
    geo.H_samples = samples
    geo.ok = True
    return geo


def scale_uncertainty_at(geo: RoadGeometry, uv_points, include_prior=True):
    """Relative 1-sigma uncertainty of along-road DISPLACEMENT between the
    nearest and farthest of the given image points (i.e. of a speed measured
    over that stretch), from the bootstrap homographies plus the cycle prior."""
    uv = np.asarray(uv_points, float)
    if geo.H is None or len(uv) < 2:
        return None
    ref = image_to_road(geo.H, uv)
    i0, i1 = int(np.argmin(ref[:, 0])), int(np.argmax(ref[:, 0]))
    d0 = np.linalg.norm(ref[i1] - ref[i0])
    if d0 < 1e-6:
        return None
    ratios = []
    for Hs in geo.H_samples:
        p = image_to_road(Hs, uv[[i0, i1]])
        ratios.append(np.linalg.norm(p[1] - p[0]) / d0)
    meas = float(np.std(ratios)) if len(ratios) > 5 else 0.05
    prior_rel = geo.prior.get("cycle_rel_sigma", 0.03) if include_prior else 0.0
    return {"measurement_rel_sigma": meas, "prior_rel_sigma": prior_rel, "total_rel_sigma": float(np.hypot(meas, prior_rel))}
