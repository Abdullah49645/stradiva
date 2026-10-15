"""
Ego-speed of a forward-facing dashcam from painted lane markings.

Why this exists
---------------
The collision pipeline (backend/pipeline) assumes a STATIC camera: it finds
vehicles by background subtraction. On a moving dashcam -- the commonest
source of accident footage -- the whole image moves, so that pipeline cannot
work at all. What a dashcam clip CAN establish, with almost no assumptions,
is the camera vehicle's own speed, which is usually the first physical fact
an investigator needs from it.

Method (OpenCV 5 throughout)
----------------------------
1. Vanishing point of the road direction per frame: Canny + probabilistic
   Hough segments below the horizon, RANSAC intersection, least-squares refit
   on inliers.
2. Candidate lane lines = dominant directions of the inlier segments around
   the vanishing point.
3. Projective resampling along each line. For ANY camera pose, along a 3D
   line the inverse of the image distance to the line's vanishing point,
   s = 1/d, is an affine function of position along the line. Sampling the
   image uniformly in s (cv2.remap of a thin strip that follows the line)
   turns the perspective-compressed dash pattern into a strictly periodic
   1D signal. Focal length, camera height, pitch and roll never need to be
   known -- they only change the affine map, which cancels below.
4. Period T_s of the dash pattern (averaged power spectrum over the window)
   and frame-to-frame shift ds of the pattern (cv2.phaseCorrelate on a
   Hanning-windowed strip). Then

        speed = dash_period_m * fps * ds / T_s

   i.e. "what fraction of one marking cycle passed per frame", times the
   cycle length. The ONLY metric input is the marking cycle length, a
   stated prior that depends on the jurisdiction (US federal 40 ft cycle,
   California freeway 48 ft cycle, ...). It is never hidden.

Closest prior art: Schoepflin & Dailey resample traffic-camera images to remove
perspective non-linearity and use autocorrelation of lane markers to calibrate
(see docs/prior_art.md). The specific contribution here is the line-wise
1/d resampling, which needs no calibration or level camera, plus the explicit
abstention and uncertainty accounting below.

Abstention (the default is to refuse, not to guess)
---------------------------------------------------
A window gets a number only if: a periodic dashed line is found (spectral
peak well above the noise floor), the per-frame-pair speeds agree with each
other, the dash period measured on the near half of the line agrees with the
far half (a wrong vanishing point or road curvature breaks this -- an
internal geometric consistency check), and enough frame pairs are valid.
Otherwise the window is returned with status ABSTAIN and the reason.

Uncertainty
-----------
Two named components, combined in quadrature on the relative scale:
  * measurement: bootstrap standard error of the window median speed;
  * prior: the marking-cycle length uncertainty (default 3 % one-sigma,
    i.e. about +-5 % at 90 %) -- this does not shrink with more frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class MarkingPrior:
    name: str
    cycle_m: float            # dash + gap
    cycle_rel_sigma: float = 0.03
    duty: float = 0.25        # dash length / cycle (10/40 ft US federal, 12/48 ft Caltrans, 3/12 m CARAI synthetic)

    @staticmethod
    def preset(key: str) -> "MarkingPrior":
        presets = {
            "us_federal": MarkingPrior("US MUTCD 10 ft + 30 ft", 12.19),
            "california_freeway": MarkingPrior("Caltrans freeway 12 ft + 36 ft", 14.63),
            "carai_synthetic": MarkingPrior("CARAI synthetic 3 m + 9 m", 12.0),
        }
        return presets[key]


@dataclass
class EgoSpeedConfig:
    n_samples: int = 512
    roi_top_frac: float = 0.40        # ignore the top of the image when looking for road segments
    min_seg_len_px: int = 18
    ransac_iters: int = 300
    ransac_angle_tol_deg: float = 1.5
    far_margin_px: float = 18.0        # do not sample closer than this to the vanishing point
    periodicity_min_snr: float = 6.0
    pair_mad_max_rel: float = 0.25
    near_far_period_max_rel: float = 0.12
    max_dash_resid_px: float = 2.0
    max_period_spread: float = 0.08
    duty_inflation: float = 0.035     # measured duty exceeds the true duty by ~0.035 (blur), synthetic dev data
    duty_tol: float = 0.06            # |measured - (prior duty + inflation)|; dev p5-p95 spread is +-0.017
    continuity_max_jump_mps: float = 6.0   # between adjacent answered windows (~2 s apart): > 3 m/s^2 sustained
    min_valid_pair_frac: float = 0.5
    max_shift_frac_of_period: float = 0.45
    n_boot: int = 400


@dataclass
class WindowResult:
    t_start_s: float
    t_end_s: float
    status: str                      # "OK" | "ABSTAIN"
    speed_mps: Optional[float] = None
    ci90_mps: Optional[List[float]] = None
    rel_sigma_measurement: Optional[float] = None
    rel_sigma_prior: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------- geometry --

def _segments(gray, cfg: EgoSpeedConfig, v_max):
    h, w = gray.shape
    y0 = int(h * cfg.roi_top_frac)
    roi = gray[y0:v_max]
    edges = cv2.Canny(cv2.GaussianBlur(roi, (5, 5), 0), 50, 150)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=cfg.min_seg_len_px, maxLineGap=8)
    if segs is None:
        return np.empty((0, 4))
    segs = segs.reshape(-1, 4).astype(float)
    segs[:, [1, 3]] += y0
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    keep = ((ang > 15) & (ang < 80)) | ((ang > 100) & (ang < 165))
    return segs[keep]


def _vanishing_point(segs, cfg: EgoSpeedConfig, rng):
    if len(segs) < 4:
        return None, np.zeros(len(segs), bool)
    p1 = np.column_stack([segs[:, :2], np.ones(len(segs))])
    p2 = np.column_stack([segs[:, 2:], np.ones(len(segs))])
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    mids = (segs[:, :2] + segs[:, 2:]) / 2.0
    dirs = (segs[:, 2:] - segs[:, :2]) / lengths[:, None]
    tol = np.radians(cfg.ransac_angle_tol_deg)

    def inliers_of(vp):
        to_vp = vp[None, :] - mids
        to_vp /= np.linalg.norm(to_vp, axis=1, keepdims=True) + 1e-9
        cosang = np.abs(np.sum(to_vp * dirs, axis=1))
        return cosang > np.cos(tol)

    best, best_score = None, -1
    for _ in range(cfg.ransac_iters):
        i, j = rng.choice(len(segs), 2, replace=False)
        x = np.cross(lines[i], lines[j])
        if abs(x[2]) < 1e-9:
            continue
        vp = x[:2] / x[2]
        inl = inliers_of(vp)
        score = lengths[inl].sum()
        if score > best_score:
            best, best_score = vp, score
    if best is None:
        return None, np.zeros(len(segs), bool)
    inl = inliers_of(best)
    if inl.sum() < 3:
        return None, inl
    # weighted least squares refit: minimise sum w (l . [x,1])^2
    A = lines[inl, :2] * lengths[inl, None]
    b = -lines[inl, 2] * lengths[inl]
    vp, *_ = np.linalg.lstsq(A, b, rcond=None)
    return vp, inliers_of(vp)


def _line_candidates(segs, inl, vp, max_lines=4):
    """Dominant line directions (angle about the VP, measured from straight
    down) among inlier segments, weighted by segment length."""
    s = segs[inl]
    if len(s) == 0:
        return []
    mids = (s[:, :2] + s[:, 2:]) / 2.0
    lengths = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
    ang = np.degrees(np.arctan2(mids[:, 0] - vp[0], mids[:, 1] - vp[1]))  # 0 = straight down
    bins = np.arange(-85, 85.5, 0.5)
    hist, _ = np.histogram(ang, bins=bins, weights=lengths)
    hist = np.convolve(hist, np.ones(3) / 3.0, mode="same")
    peaks = []
    h = hist.copy()
    for _ in range(max_lines):
        k = int(np.argmax(h))
        if h[k] <= 0:
            break
        peaks.append(float((bins[k] + bins[k + 1]) / 2.0))
        h[max(0, k - 6): k + 7] = 0  # suppress +-3 deg
    return peaks


def _strip_maps(vp, angle_deg, v_limit, w, h, cfg: EgoSpeedConfig, half_width_frac=0.035, n_perp=9):
    """Sampling grid that follows a line from the VP downward, uniform in
    s = 1/d. Returns (map_x, map_y, s_values, d_values) or None."""
    a = np.radians(angle_deg)
    direction = np.array([np.sin(a), np.cos(a)])          # pointing away from VP, downward
    normal = np.array([direction[1], -direction[0]])
    # farthest usable distance from VP along the line: stay inside image and above v_limit
    ts = []
    for bound, comp in ((w - 1, 0), (0, 0), (v_limit - 1, 1)):
        if abs(direction[comp]) > 1e-9:
            t = (bound - vp[comp]) / direction[comp]
            if t > 0:
                ts.append(t)
    if not ts:
        return None
    d_max = min(ts)
    d_min = max(cfg.far_margin_px, d_max / 12.0)
    if d_max <= d_min * 2:
        return None
    s = np.linspace(1.0 / d_min, 1.0 / d_max, cfg.n_samples)  # far -> near
    d = 1.0 / s
    offs = np.linspace(-1, 1, n_perp)
    centre = vp[None, :] + d[:, None] * direction[None, :]
    half = np.maximum(1.5, half_width_frac * d)
    pts = centre[None, :, :] + (offs[:, None, None] * half[None, :, None]) * normal[None, None, :]
    return pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32), s, d


def _profile(gray, maps):
    mx, my, s, d = maps
    strip = cv2.remap(gray, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    prof = strip.astype(np.float32).max(axis=0)
    # high-pass: remove slow illumination trends along the line
    trend = cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), sigmaX=len(prof) / 16.0).ravel()
    p = prof - trend
    sd = p.std()
    return (p / sd) if sd > 1e-6 else p


def _raw_profile(gray, maps):
    mx, my, s, d = maps
    strip = cv2.remap(gray, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return strip.astype(np.float32).max(axis=0)


def _normalise(p):
    q = p - p.mean()
    sd = q.std()
    return q / sd if sd > 1e-6 else q


def _side_level(gray, vp, ang, v_limit, w, h, cfg):
    vals = []
    for off in (-3.0, 3.0):
        m = _strip_maps(vp, ang + off, v_limit, w, h, cfg)
        if m is not None:
            mx, my, s, d = m
            strip = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            v = strip[strip > 0]
            if v.size:
                vals.append(float(np.median(v)))
    return float(np.median(vals)) if vals else None


def _dash_centres(prof, s_vals, asphalt, return_lengths=False):
    lo = asphalt if asphalt is not None else np.percentile(prof, 30)
    hi = np.percentile(prof, 97)
    if hi - lo < 15:
        return ([], []) if return_lengths else []
    thr = 0.5 * (lo + hi)
    above = prof > thr
    cen, lens, i, n = [], [], 0, len(prof)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            if i > 0 and j < n - 1:                     # complete dashes only
                fi = (i - 1) + (thr - prof[i - 1]) / (prof[i] - prof[i - 1] + 1e-9)
                fj = j + (prof[j] - thr) / (prof[j] - prof[j + 1] + 1e-9)
                cen.append(float(np.interp(0.5 * (fi + fj), np.arange(n), s_vals)))
                lens.append(float(abs(np.interp(fj, np.arange(n), s_vals) - np.interp(fi, np.arange(n), s_vals))))
            i = j + 1
        else:
            i += 1
    order = np.argsort(cen)
    if return_lengths:
        return [cen[k] for k in order], [lens[k] for k in order]
    return [cen[k] for k in order]


def _period_from_centres(cen):
    """Base spacing of dash centres in s (tolerates a missing dash), and the
    RMS residual of the centres from an equally spaced pattern, in pixels."""
    cen = np.asarray(cen)
    gaps = np.diff(cen)
    if len(gaps) == 0:
        return None, None
    g = np.sort(gaps)
    base = np.median(g[: max(1, len(g) // 2 + 1)])
    mult = np.maximum(1, np.round(gaps / base))
    base = float(np.sum(gaps) / np.sum(mult))
    k = np.concatenate([[0], np.cumsum(np.round(gaps / base))])
    A = np.column_stack([np.ones_like(k), k])
    wts = 1.0 / np.maximum(cen, 1e-9) ** 2
    coef, *_ = np.linalg.lstsq(A * wts[:, None], cen * wts, rcond=None)
    r_px = (cen - A @ coef) / np.maximum(cen, 1e-9) ** 2
    return float(abs(coef[1])), float(np.sqrt(np.mean(r_px ** 2)))


def _period_from_profiles(profiles):
    """Averaged power spectrum peak. Returns (period_in_samples, snr)."""
    P = np.array(profiles)
    n = P.shape[1]
    win = np.hanning(n)
    spec = np.mean(np.abs(np.fft.rfft(P * win[None, :], axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(n)
    valid = (freqs > 2.0 / n) & (freqs < 0.25)
    if valid.sum() < 5:
        return None, 0.0
    idx = np.where(valid)[0]
    k = idx[np.argmax(spec[idx])]
    # parabolic refinement
    if 1 <= k < len(spec) - 1:
        y0, y1, y2 = np.log(spec[k - 1] + 1e-12), np.log(spec[k] + 1e-12), np.log(spec[k + 1] + 1e-12)
        delta = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) != 0 else 0.0
    else:
        delta = 0.0
    f = (k + delta) / n
    noise = np.median(spec[idx])
    snr = float(spec[k] / (noise + 1e-12))
    return (1.0 / f if f > 0 else None), snr


def _shift(p0, p1, win2d):
    a = np.tile(p0.astype(np.float32), (16, 1))
    b = np.tile(p1.astype(np.float32), (16, 1))
    (dx, _dy), resp = cv2.phaseCorrelate(a, b, win2d)
    return dx, resp


# ------------------------------------------------------------- per window --

def estimate_window(frames_gray, fps, t_start_s, prior: MarkingPrior, cfg: EgoSpeedConfig = None,
                    v_limit=None, rng=None):
    cfg = cfg or EgoSpeedConfig()
    rng = rng or np.random.default_rng(0)
    n = len(frames_gray)
    h, w = frames_gray[0].shape
    res = WindowResult(t_start_s=t_start_s, t_end_s=t_start_s + n / fps, status="ABSTAIN")
    v_limit = v_limit or h

    # 1) VP per frame, robust median over the window
    vps, cand_angles = [], []
    for g in frames_gray[:: max(1, n // 10)]:
        segs = _segments(g, cfg, v_limit)
        vp, inl = _vanishing_point(segs, cfg, rng)
        if vp is not None and 0 <= vp[1] < h:
            vps.append(vp)
            cand_angles.extend(_line_candidates(segs, inl, vp))
    if len(vps) < 3:
        res.reasons.append("no stable road vanishing point (lane lines not found)")
        return res
    vps = np.array(vps)
    vp = np.median(vps, axis=0)
    vp_spread = float(np.median(np.abs(vps - vp)))
    res.diagnostics.update({"vp": vp.tolist(), "vp_mad_px": vp_spread})

    # cluster candidate angles across sampled frames
    if not cand_angles:
        res.reasons.append("no lane-line directions found")
        return res
    cand_angles = np.sort(np.array(cand_angles))
    clusters = []
    for a in cand_angles:
        if clusters and abs(a - np.mean(clusters[-1])) < 2.0:
            clusters[-1].append(a)
        else:
            clusters.append([a])
    lines = [float(np.median(c)) for c in clusters if len(c) >= max(2, len(vps) // 3)]
    if not lines:
        res.reasons.append("lane-line directions not consistent across the window")
        return res

    # 2) per line: dash segmentation in s = 1/d (period from dash-centre spacing, as for the static
    #    camera), frame-to-frame shift from phase correlation of the resampled profiles
    win2d = cv2.createHanningWindow((cfg.n_samples, 16), cv2.CV_32F)
    line_results = []
    for ang in lines:
        maps = _strip_maps(vp, ang, v_limit, w, h, cfg)
        if maps is None:
            continue
        s_vals = maps[2]
        raw = [_raw_profile(g, maps) for g in frames_gray]
        side = _side_level(frames_gray[len(frames_gray) // 2], vp, ang, v_limit, w, h, cfg)
        periods, resids, n_d, duties = [], [], [], []
        for prof in raw:
            cen, lens = _dash_centres(prof, s_vals, side, return_lengths=True)
            if len(cen) < 2:
                continue
            T, rpx = _period_from_centres(cen)
            if T is not None:
                periods.append(T)
                # duty from the NEAREST complete dash (smallest s): least blur inflation in s
                duties.append(lens[0] / T)
                if len(cen) >= 3:
                    resids.append(rpx)       # spacing residual only testable with >= 3 dashes
                n_d.append(len(cen))
        frac_frames = len(periods) / max(1, len(raw))
        if frac_frames < 0.5:
            line_results.append({"angle_deg": ang, "dashed": False, "frames_with_dashes": frac_frames})
            continue
        T_s = float(np.median(periods))                       # period in s units (s decreases towards the camera)
        T_idx = T_s / abs(s_vals[1] - s_vals[0])              # period in profile samples
        profs = [_normalise(p) for p in raw]
        shifts = []
        for p0, p1 in zip(profs[:-1], profs[1:]):
            dx, _r = _shift(p0, p1, win2d)
            shifts.append(dx)
        shifts = np.array(shifts)
        valid = np.abs(shifts) < cfg.max_shift_frac_of_period * T_idx
        speeds = prior.cycle_m * fps * shifts[valid] / T_idx
        pair_t = (np.arange(len(shifts))[valid] + 0.5) / fps
        line_results.append({"angle_deg": ang, "dashed": True, "frames_with_dashes": frac_frames,
                             "period_samples": T_idx,
                             "period_rel_spread": float(1.4826 * np.median(np.abs(np.array(periods) - T_s)) / T_s),
                             "dash_resid_px_median": float(np.median(resids)) if resids else 0.0,
                             "frames_with_3_dashes": len(resids) / max(1, len(raw)), "speeds": speeds, "pair_t": pair_t,
                             "duty_measured": float(np.median(duties)) if duties else None,
                             "valid_pair_frac": float(valid.mean()) if len(valid) else 0.0,
                             "near_far_period_rel_diff": float(np.median(resids)) if resids else None})

    dashed = [lr for lr in line_results if lr.get("dashed")]
    res.diagnostics["lines"] = [{k: (v if k != "speeds" else None) for k, v in lr.items()} for lr in line_results]
    if not dashed:
        res.reasons.append("no periodic (dashed) lane marking found -- metric scale unavailable")
        return res

    # 3) quality gates per dashed line, then combine the lines that pass
    passing = []
    for lr in dashed:
        why = []
        sp = lr["speeds"]
        if lr["valid_pair_frac"] < cfg.min_valid_pair_frac or len(sp) < 5:
            why.append("too few valid frame pairs")
        else:
            # residual spread around a robust linear trend (the camera may genuinely accelerate)
            tp = lr["pair_t"]
            slope = np.median([(sp[j] - sp[i]) / (tp[j] - tp[i]) for i in range(len(sp)) for j in range(i + 1, len(sp))
                               if tp[j] > tp[i]]) if len(sp) > 2 else 0.0
            inter = np.median(sp - slope * tp)
            resid_tr = sp - (inter + slope * tp)
            mad = float(np.median(np.abs(resid_tr))) * 1.4826
            scale = max(abs(float(np.median(sp))), 1.0)
            lr["trend_mps2"] = float(slope)
            if mad / scale > cfg.pair_mad_max_rel:
                why.append(f"frame-pair speeds disagree (robust spread {mad/scale:.0%} around the trend)")
        if lr.get("duty_measured") is not None and abs(lr["duty_measured"] - prior.duty - cfg.duty_inflation) > cfg.duty_tol:
            why.append(f"marking proportions differ from the assumed standard (dash covers {lr['duty_measured']:.0%} of the "
                       f"cycle, standard {prior.duty:.0%}): wrong standard or not lane markings (e.g. pavement joints)")
        if lr["frames_with_3_dashes"] < 0.2:
            why.append("too few frames show three complete dashes to test the equal-spacing assumption")
        elif lr["dash_resid_px_median"] > cfg.max_dash_resid_px:
            why.append(f"dash centres are not equally spaced along the line (median residual "
                       f"{lr['dash_resid_px_median']:.1f}px): vanishing point or straight-road assumption violated")
        if lr["period_rel_spread"] > cfg.max_period_spread:
            why.append(f"marking period unstable across frames ({lr['period_rel_spread']:.0%})")
        lr["gate_failures"] = why
        if not why:
            passing.append(lr)
    if not passing:
        res.reasons.extend(sorted({r for lr in dashed for r in lr["gate_failures"]}))
        return res

    all_speeds = np.concatenate([lr["speeds"] for lr in passing])
    if len(passing) > 1:
        meds = [np.median(lr["speeds"]) for lr in passing]
        if max(meds) - min(meds) > 0.1 * max(abs(np.mean(meds)), 1.0):
            res.reasons.append("two dashed lines give different speeds (>10 %)")
            return res
    med = float(np.median(all_speeds))
    boots = [np.median(rng.choice(all_speeds, len(all_speeds))) for _ in range(cfg.n_boot)]
    se = float(np.std(boots))
    scale = max(abs(med), 0.5)
    rel_meas = se / scale
    rel_prior = prior.cycle_rel_sigma
    rel = float(np.hypot(rel_meas, rel_prior))
    half = 1.645 * rel * scale
    res.status = "OK"
    res.speed_mps = med
    res.ci90_mps = [max(0.0, med - half), med + half]
    res.rel_sigma_measurement = rel_meas
    res.rel_sigma_prior = rel_prior
    res.diagnostics["n_pairs"] = int(len(all_speeds))
    return res


def detect_static_bottom(frames_gray, max_frac=0.35, std_thresh=4.0):
    """Rows at the bottom of the frame that do not change over time (bonnet /
    dashboard). Returns the first usable row index from the bottom."""
    stack = np.stack(frames_gray[:: max(1, len(frames_gray) // 20)]).astype(np.float32)
    row_std = stack.std(axis=0).mean(axis=1)
    h = len(row_std)
    v = h
    while v > h * (1 - max_frac) and row_std[v - 1] < std_thresh:
        v -= 1
    return v - 4 if v < h else h


def estimate_ego_speed(video_path, prior: MarkingPrior, window_s=2.0, cfg: EgoSpeedConfig = None,
                       max_frames=None, frame_stride=1, fps_override=None, continuity_check=True):
    """fps_override: the TRUE recording rate if known (sensor timestamps, device
    specification). Container fps metadata is frequently wrong for raw dashcam
    streams; every speed scales linearly with it, so its source is reported."""
    cfg = cfg or EgoSpeedConfig()
    cap = cv2.VideoCapture(video_path)
    fps_container = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_override or fps_container or 30.0
    fps_source = "override (supplied by the user / sensor timestamps)" if fps_override else "container metadata (unverified)"
    eff_fps = fps / frame_stride
    win_n = int(round(window_s * eff_fps))
    results, buf, k, start_idx = [], [], 0, 0
    v_limit = None
    rng = np.random.default_rng(0)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % frame_stride == 0:
            buf.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        k += 1
        if len(buf) == win_n:
            if v_limit is None:
                v_limit = detect_static_bottom(buf)
            results.append(estimate_window(buf, eff_fps, start_idx / fps, prior, cfg, v_limit=v_limit, rng=rng))
            start_idx = k
            buf = []
        if max_frames and k >= max_frames:
            break
    cap.release()
    if continuity_check:
        _continuity(results, cfg)
    return {"fps": fps, "fps_container": fps_container, "fps_source": fps_source, "prior": asdict(prior),
            "window_s": window_s, "v_limit": v_limit, "windows": [asdict(r) for r in results]}


def _continuity(results, cfg):
    """An answered window whose speed jumps by more than continuity_max_jump_mps
    relative to BOTH answered neighbours (when it has any within 4 s) is
    physically implausible (sustained > ~3 m/s^2 then reversed) and is withdrawn."""
    ok = [i for i, r in enumerate(results) if r.status == "OK"]
    flag = []
    for k, i in enumerate(ok):
        nb = [j for j in (ok[k - 1] if k > 0 else None, ok[k + 1] if k + 1 < len(ok) else None)
              if j is not None and abs(results[j].t_start_s - results[i].t_start_s) <= 4.01]
        if nb and all(abs(results[i].speed_mps - results[j].speed_mps) > cfg.continuity_max_jump_mps for j in nb):
            flag.append(i)
    for i in flag:
        r = results[i]
        r.diagnostics["withdrawn_speed_mps"] = r.speed_mps
        r.status, r.speed_mps, r.ci90_mps = "ABSTAIN", None, None
        r.reasons.append("speed inconsistent with neighbouring windows (physically implausible jump)")
