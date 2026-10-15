"""
Projective road-geometry primitives shared by the static-camera geometry
(backend/geometry/road_geometry.py) and the dashcam ego-speed estimator
(backend/dashcam/ego_speed.py). OpenCV does the image work (Canny,
HoughLinesP, remap, phaseCorrelate); the projective reasoning is spelled out
here so it can be checked by hand.

Two facts carry most of the weight:

1. Along any 3D line, the inverse image distance to the line's vanishing
   point, s = 1/d, is an AFFINE function of position along the line
   (whatever the focal length, pitch, roll or yaw). Painted dashes, equally
   spaced on the road, are therefore equally spaced in s.

2. For three coplanar parallel lines at lateral offsets -W, 0, +W (two road
   edges and the centre line), the image of the road plane's vanishing line
   (the horizon) is the harmonic conjugate of the centre line with respect
   to the two edges in the pencil of lines through their vanishing point.
   The horizon's tilt is the camera roll -- recovered in closed form, no
   search over candidate angles.
"""
from __future__ import annotations

import cv2
import numpy as np


def line_segments(gray, roi_top_frac=0.0, v_max=None, min_len=18, canny=(50, 150), hough_thresh=30, max_gap=8,
                  min_angle_deg=12.0):
    h, w = gray.shape[:2]
    y0 = int(h * roi_top_frac)
    y1 = v_max or h
    roi = gray[y0:y1]
    edges = cv2.Canny(cv2.GaussianBlur(roi, (5, 5), 0), canny[0], canny[1])
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_thresh, minLineLength=min_len, maxLineGap=max_gap)
    if segs is None:
        return np.empty((0, 4))
    segs = segs.reshape(-1, 4).astype(float)
    segs[:, [1, 3]] += y0
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    keep = (ang > min_angle_deg) & (ang < 180 - min_angle_deg)
    return segs[keep]


def vanishing_point_ransac(segs, rng, iters=300, angle_tol_deg=1.5):
    """RANSAC intersection of line segments, weighted least-squares refit on
    inliers. Returns (vp_xy or None, inlier_mask)."""
    if len(segs) < 3:
        return None, np.zeros(len(segs), bool)
    p1 = np.column_stack([segs[:, :2], np.ones(len(segs))])
    p2 = np.column_stack([segs[:, 2:], np.ones(len(segs))])
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    mids = (segs[:, :2] + segs[:, 2:]) / 2.0
    dirs = (segs[:, 2:] - segs[:, :2]) / lengths[:, None]
    cos_tol = np.cos(np.radians(angle_tol_deg))

    def inliers_of(vp):
        to_vp = vp[None, :] - mids
        to_vp /= np.linalg.norm(to_vp, axis=1, keepdims=True) + 1e-9
        return np.abs(np.sum(to_vp * dirs, axis=1)) > cos_tol

    best, best_score = None, -1.0
    n = len(segs)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        x = np.cross(lines[i], lines[j])
        if abs(x[2]) < 1e-9:
            continue
        vp = x[:2] / x[2]
        score = lengths[inliers_of(vp)].sum()
        if score > best_score:
            best, best_score = vp, score
    if best is None:
        return None, np.zeros(n, bool)
    inl = inliers_of(best)
    if inl.sum() < 3:
        return None, inl
    A = lines[inl, :2] * lengths[inl, None]
    b = -lines[inl, 2] * lengths[inl]
    vp, *_ = np.linalg.lstsq(A, b, rcond=None)
    return vp, inliers_of(vp)


def line_directions_about_vp(segs, inl, vp, max_lines=6, sep_deg=3.0):
    """Dominant directions (deg, 0 = straight down from the VP, positive =
    towards image right) of inlier segments, length-weighted."""
    s = segs[inl]
    if len(s) == 0:
        return []
    mids = (s[:, :2] + s[:, 2:]) / 2.0
    lengths = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
    ang = np.degrees(np.arctan2(mids[:, 0] - vp[0], mids[:, 1] - vp[1]))
    bins = np.arange(-88, 88.25, 0.25)
    hist, _ = np.histogram(ang, bins=bins, weights=lengths)
    hist = np.convolve(hist, np.ones(5) / 5.0, mode="same")
    peaks, h = [], hist.copy()
    half = int(sep_deg / 0.25)
    for _ in range(max_lines):
        k = int(np.argmax(h))
        if h[k] <= 0.05 * hist.max():
            break
        # weighted mean angle of segments near the peak (sub-bin precision)
        centre = (bins[k] + bins[k + 1]) / 2.0
        near = np.abs(ang - centre) < 1.0
        peaks.append({"angle_deg": float(np.average(ang[near], weights=lengths[near])) if near.any() else float(centre),
                      "support_px": float(lengths[near].sum())})
        h[max(0, k - half): k + half + 1] = 0
    return peaks


def direction_vector(angle_deg):
    a = np.radians(angle_deg)
    return np.array([np.sin(a), np.cos(a)])


def harmonic_horizon_direction(vp, ang_left, ang_centre, ang_right, probe_dist=200.0):
    """Unit image direction of the horizon (vanishing line of the road plane)
    through `vp`, as the harmonic conjugate of the centre line w.r.t. the two
    edge lines (world offsets -W, 0, +W). Uses a probe transversal
    perpendicular to the centre line at distance probe_dist below the VP."""
    dc = direction_vector(ang_centre)
    nrm = np.array([dc[1], -dc[0]])          # transversal direction (perpendicular to centre line)
    p0 = vp + probe_dist * dc                  # transversal passes through here

    def coord(angle):
        d = direction_vector(angle)
        # intersect vp + t d with p0 + u nrm  ->  solve [d, -nrm] [t,u]^T = p0 - vp
        M = np.column_stack([d, -nrm])
        t, u = np.linalg.solve(M, p0 - vp)
        return u

    a, c, b = coord(ang_left), 0.0, coord(ang_right)
    den = 2 * c - a - b
    if abs(den) < 1e-9:
        return nrm / np.linalg.norm(nrm)       # horizon parallel to the transversal
    d_u = (a * c + b * c - 2 * a * b) / den
    hp = p0 + d_u * nrm                        # point on the horizon line (on the transversal)
    hd = hp - vp
    hd /= np.linalg.norm(hd)
    if hd[0] < 0:
        hd = -hd
    return hd


def intersect_lines(p, d, q, e):
    """Intersection of p + t d and q + u e (2D)."""
    M = np.column_stack([d, -e])
    if abs(np.linalg.det(M)) < 1e-12:
        return None
    t, _ = np.linalg.solve(M, q - p)
    return p + t * d


def strip_along_line(vp, angle_deg, d_min, d_max, n_samples=512, half_width_frac=0.03, n_perp=7, min_half_px=1.5):
    """Sampling grid following the line from the VP outwards, uniform in
    s = 1/d. Returns (map_x, map_y, s, d)."""
    direction = direction_vector(angle_deg)
    normal = np.array([direction[1], -direction[0]])
    s = np.linspace(1.0 / d_min, 1.0 / d_max, n_samples)   # far -> near
    d = 1.0 / s
    offs = np.linspace(-1, 1, n_perp)
    centre = vp[None, :] + d[:, None] * direction[None, :]
    half = np.maximum(min_half_px, half_width_frac * d)
    pts = centre[None, :, :] + (offs[:, None, None] * half[None, :, None]) * normal[None, None, :]
    return pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32), s, d


def min_distance_inside(vp, angle_deg, w, h, margin=2.0):
    """Distance from the VP along the line at which it enters the image (0 if
    the VP itself is inside)."""
    direction = direction_vector(angle_deg)
    t_in = 0.0
    for comp, lo, hi in ((0, 0.0, w - 1.0), (1, 0.0, h - 1.0)):
        if vp[comp] < lo and direction[comp] > 1e-9:
            t_in = max(t_in, (lo - vp[comp]) / direction[comp])
        elif vp[comp] > hi and direction[comp] < -1e-9:
            t_in = max(t_in, (hi - vp[comp]) / direction[comp])
    return t_in + margin if t_in > 0 else 0.0


def max_distance_inside(vp, angle_deg, w, h, v_limit=None):
    direction = direction_vector(angle_deg)
    v_limit = v_limit or h
    ts = []
    for bound, comp in ((w - 1, 0), (0, 0), (v_limit - 1, 1)):
        if abs(direction[comp]) > 1e-9:
            t = (bound - vp[comp]) / direction[comp]
            if t > 0:
                ts.append(t)
    return min(ts) if ts else None


def sample_strip(gray, maps, reduce="max"):
    mx, my, s, d = maps
    strip = cv2.remap(gray, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    strip = strip.astype(np.float32)
    return strip.max(axis=0) if reduce == "max" else strip.mean(axis=0)
