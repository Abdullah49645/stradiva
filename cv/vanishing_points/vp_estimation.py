"""
Stage: vanishing-point estimation (brief section 9).

Given robust x=m*y+b fits for the left and right lane edges, the
vanishing point is their intersection. We also report a simple
stability metric: how much the VP estimate moves if we fit using only
the top half vs bottom half of each line's inlier points (a stand-in
for "how stable is this VP across the observed geometry" ahead of
building a full multi-frame stability check in Week 2).
"""
import numpy as np


def intersect_x_of_y_lines(line1, line2):
    """
    line: (m, b) with x = m*y + b.
    Returns (u, v) intersection point, or None if parallel.
    """
    (m1, b1), (m2, b2) = line1, line2
    if abs(m1 - m2) < 1e-9:
        return None
    v = (b2 - b1) / (m1 - m2)
    u = m1 * v + b1
    return float(u), float(v)


def estimate_vp_stability(segs_left, segs_right, fit_fn, n_splits=3):
    """
    Refit left/right lines from random half-subsets of the segment pool
    and measure the spread of the resulting VP estimate. Returns
    (vp_mean, vp_std_px) as a crude stability/confidence signal.
    """
    rng = np.random.default_rng(0)
    vps = []
    for _ in range(n_splits):
        if len(segs_left) >= 2:
            idx_l = rng.choice(len(segs_left), size=max(2, len(segs_left) // 2), replace=False)
        else:
            idx_l = np.arange(len(segs_left))
        if len(segs_right) >= 2:
            idx_r = rng.choice(len(segs_right), size=max(2, len(segs_right) // 2), replace=False)
        else:
            idx_r = np.arange(len(segs_right))
        fl = fit_fn(segs_left[idx_l]) if len(idx_l) else None
        fr = fit_fn(segs_right[idx_r]) if len(idx_r) else None
        if fl and fr:
            vp = intersect_x_of_y_lines(fl, fr)
            if vp:
                vps.append(vp)
    if not vps:
        return None, None
    vps = np.array(vps)
    return vps.mean(axis=0), vps.std(axis=0).mean()
