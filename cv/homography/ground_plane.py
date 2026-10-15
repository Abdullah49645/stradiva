"""
Stage: ground-plane homography / scale recovery (brief sections 11-12).

Key geometric fact exploited here (true for any camera with zero roll,
looking down a straight road): the image row (v-coordinate) of a ground
point depends only on its world forward-distance X and camera
height/pitch -- NOT on its lateral offset Y. So every point sharing an
image row shares a world X. That means: wherever we can pin down world
X at a given row (from a standardized road marking, e.g. a lane-dash
of known length/gap), we get world X for *every* point in that row,
including the lane edges. Combined with the known lane width (world Y
= +-1.85 m) this yields (image, world) point correspondences without
ever needing the vehicle or ground-truth data -- only the painted lane
geometry, exactly as the brief specifies as the intended scale
strategy (sections 12, "standardized road markings", "lane width").

This module intentionally does NOT use any ground-truth camera
parameters. It only uses:
  - fitted left/right edge lines (from cv/lines)
  - detected dash blobs on the road surface
  - the *assumed* standard dash pattern (a configurable prior, flagged
    as a scale-source/uncertainty input downstream)
  - the *assumed* standard lane width (also a flagged prior)
"""
import cv2
import numpy as np


def detect_dash_rows(bgr, left_line, right_line, white_thresh=200, band_frac=0.3):
    """
    Scan image rows, look for near-white pixels near the lane centerline
    (halfway between the fitted left/right edges at that row), and group
    contiguous "white" rows into dash blobs.

    Returns list of dicts: {v_near, v_far, u_center_near, u_center_far}
    (near = larger v i.e. closer to camera / bottom of image).
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ml, bl = left_line
    mr, br = right_line

    row_is_dash = np.zeros(h, dtype=bool)
    center_u = np.full(h, np.nan)
    for v in range(h):
        ul = ml * v + bl
        ur = mr * v + br
        if not (0 <= ul < w and 0 <= ur < w) or ur <= ul:
            continue
        cu = (ul + ur) / 2.0
        half = max(8, int((ur - ul) * 0.06))  # narrow window around centerline
        u0, u1 = int(cu - half), int(cu + half)
        u0, u1 = max(0, u0), min(w, u1)
        if u1 <= u0:
            continue
        window = gray[v, u0:u1]
        if window.size and np.mean(window > white_thresh) > band_frac:
            row_is_dash[v] = True
            center_u[v] = cu

    # group contiguous True rows into blobs
    dashes = []
    v = 0
    while v < h:
        if row_is_dash[v]:
            v_start = v
            while v < h and row_is_dash[v]:
                v += 1
            v_end = v - 1  # inclusive
            dashes.append({
                "v_near": v_end,     # larger v = nearer camera
                "v_far": v_start,    # smaller v = farther
                "u_near": float(center_u[v_end]),
                "u_far": float(center_u[v_start]),
            })
        else:
            v += 1
    return dashes


def build_correspondences(dashes, left_line, right_line, lane_half_width_m,
                           dash_len_m, dash_gap_m, min_dash_span_px=3):
    """
    Assign world X to each detected dash assuming they are the 0th,1st,...
    dash counting from nearest-to-camera, per the standard pattern, then
    emit (image_uv, world_xy) correspondences at the near/far edge of each
    dash for the center line and, at the same rows, the left/right edges.
    """
    period = dash_len_m + dash_gap_m
    ml, bl = left_line
    mr, br = right_line

    # keep plausible dashes only (reject noise slivers)
    dashes = [d for d in dashes if (d["v_near"] - d["v_far"]) >= min_dash_span_px]
    # nearest dash (largest v_near) first
    dashes = sorted(dashes, key=lambda d: -d["v_near"])

    img_pts, world_pts = [], []
    for k, d in enumerate(dashes):
        x_near = k * period
        x_far = k * period + dash_len_m
        for v, x_world in ((d["v_near"], x_near), (d["v_far"], x_far)):
            u_left = ml * v + bl
            u_right = mr * v + br
            u_center = (u_left + u_right) / 2.0
            img_pts += [(u_left, v), (u_center, v), (u_right, v)]
            world_pts += [
                (x_world, -lane_half_width_m),
                (x_world, 0.0),
                (x_world, lane_half_width_m),
            ]
    return np.array(img_pts, dtype=np.float32), np.array(world_pts, dtype=np.float32)


def fit_ground_homography(img_pts, world_pts):
    """
    image (u,v) -> world (X,Y) homography, RANSAC-robust to bad dash
    detections. Returns (H, inlier_mask) or (None, None) on failure.
    """
    if len(img_pts) < 4:
        return None, None
    H, mask = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, ransacReprojThreshold=0.3)
    return H, mask


def image_to_world(H, uv_points):
    pts = np.atleast_2d(np.asarray(uv_points, dtype=np.float32))
    pts_h = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    return pts_h


def world_to_image(H, world_points):
    """
    Inverse of image_to_world: project world (X,Y) ground points back to
    image (u,v) pixels. Used by the evidence-video overlay to draw
    physically-meaningful markers (e.g. a "required stopping point")
    directly on the road at their true reconstructed location.
    """
    H_inv = np.linalg.inv(H)
    pts = np.atleast_2d(np.asarray(world_points, dtype=np.float32))
    pts_h = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
    return pts_h


def lane_width_self_consistency(H, left_line, right_line, image_h, assumed_width_m,
                                 n_samples=12, v_margin=0.15):
    """
    Independent sanity check (does NOT catch the dash-anchoring offset --
    see docs/week1_feasibility.md -- but DOES catch scale/skew errors):
    reproject the fitted left/right edge lines through the homography at
    several rows and check the recovered lane width is consistent with the
    assumed constant, and consistent with *itself* across depth (a sheared
    or mis-scaled homography will make width drift with distance even if
    it happens to match at one row).

    Returns dict: {mean_width_m, std_width_m, max_abs_error_m, consistent}
    """
    ml, bl = left_line
    mr, br = right_line
    v_lo, v_hi = int(image_h * v_margin), int(image_h * (1 - v_margin))
    vs = np.linspace(v_lo, v_hi, n_samples)
    widths = []
    for v in vs:
        ul, ur = ml * v + bl, mr * v + br
        world = image_to_world(H, [[ul, v], [ur, v]])
        widths.append(abs(world[1, 1] - world[0, 1]))
    widths = np.array(widths)
    mean_w, std_w = float(widths.mean()), float(widths.std())
    max_err = float(np.max(np.abs(widths - assumed_width_m)))
    # "consistent" = both close to the assumed constant AND stable across depth
    consistent = bool(max_err < 0.5 and std_w < 0.25)
    return {
        "mean_width_m": mean_w, "std_width_m": std_w,
        "max_abs_error_vs_assumed_m": max_err, "consistent": consistent,
    }
