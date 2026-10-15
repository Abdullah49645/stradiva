"""
Stage: scene line detection (brief section 9).

Canny -> probabilistic Hough transform -> segments, with a light
robustness pass (filter by length/angle) since raw Hough output is noisy.
"""
import cv2
import numpy as np


def detect_line_segments(gray_or_bgr, canny_lo=50, canny_hi=150,
                          hough_thresh=40, min_len=40, max_gap=15):
    if gray_or_bgr.ndim == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_thresh,
                            minLineLength=min_len, maxLineGap=max_gap)
    if segs is None:
        return np.empty((0, 4)), edges
    return segs.reshape(-1, 4).astype(float), edges


def segment_angle_deg(seg):
    x1, y1, x2, y2 = seg
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def cluster_by_slope_side(segs, img_w, angle_tol_deg=25):
    """
    Split near-vertical segments (the two lane edges converge steeply)
    into 'left' and 'right' clusters by their x-position and slope sign.
    Filters out near-horizontal noise (angle close to 0/180).
    """
    left, right = [], []
    for seg in segs:
        x1, y1, x2, y2 = seg
        ang = segment_angle_deg(seg)
        # keep only steep-ish segments (near vertical), reject near-horizontal
        steepness = min(abs(ang - 90), abs(ang - 90))
        if abs(ang - 90) > (90 - 20):  # too close to horizontal
            continue
        mid_x = (x1 + x2) / 2.0
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < 1e-6:
            continue
        slope = dx / dy  # how much x changes per unit y (near-vertical lines)
        if slope < 0 and mid_x < img_w * 0.65:
            left.append(seg)
        elif slope > 0 and mid_x > img_w * 0.35:
            right.append(seg)
    return np.array(left) if left else np.empty((0, 4)), \
        np.array(right) if right else np.empty((0, 4))


def fit_line_x_of_y(segs):
    """
    Robustly fit x = m*y + b through a cluster of segments (near-vertical
    lines are better parameterized this way than y = m*x + b).
    Returns (m, b) or None if insufficient data.
    """
    if len(segs) == 0:
        return None
    pts = []
    for x1, y1, x2, y2 in segs:
        pts.append((x1, y1))
        pts.append((x2, y2))
    pts = np.array(pts)
    ys, xs = pts[:, 1], pts[:, 0]
    # cv2.fitLine with robust (Huber) distance type
    line_pts = np.stack([ys, xs], axis=1).astype(np.float32)
    vy, vx, y0, x0 = cv2.fitLine(line_pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if abs(vy) < 1e-9:
        return None
    m = vx / vy
    b = x0 - m * y0
    return float(m), float(b)
