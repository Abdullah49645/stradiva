"""
Controlled synthetic test-scene generator for Odolite Week-1 feasibility.

Why synthetic: Week 1's job is to prove the *reconstruction math* works
end-to-end (lines -> vanishing point -> scale-anchored homography ->
tracking -> trajectory -> speed) before trusting it on real accident
footage. A synthetic scene gives us exact ground truth (camera pose,
road geometry, vehicle speed) to check the pipeline against, exactly as
prescribed in section 24 ("create your own controlled ground-truth
dataset") and section 38 ("create a controlled test scene").

Camera model
------------
World frame: X = distance down the road (forward), Y = lateral offset,
Z = height above ground (road plane is Z=0).

Camera sits at world position (0, 0, H), pitched down by angle THETA
(radians) from horizontal, no roll/yaw. Camera axes expressed in world
coordinates:
    forward = (cos(theta), 0, -sin(theta))
    down    = (-sin(theta), 0, -cos(theta))
    right   = (0, 1, 0)

For world point P, camera-space depth/right/down are dot products of
(P - C) with these axes; image coords follow the standard pinhole model
u = cx + f*Xc/Zc, v = cy + f*Yc/Zc. This is a plain, independently
checkable rotation -- not borrowed from any CV library -- so ground
truth is trustworthy.
"""
import json
import math
import os

import cv2
import numpy as np

# ----------------------------- scene / camera ground truth -----------------
IMG_W, IMG_H = 1280, 720
FOCAL_PX = 950.0
CX, CY = IMG_W / 2.0, IMG_H / 2.0
CAM_HEIGHT_M = 6.0          # elevated CCTV-like mount
PITCH_DEG = 22.0            # downward tilt from horizontal
LANE_HALF_WIDTH_M = 1.85    # -> 3.7 m lane, US standard
DASH_LEN_M = 3.0
DASH_GAP_M = 9.0
DASH_PERIOD_M = DASH_LEN_M + DASH_GAP_M
ROAD_X_MAX_M = 160.0

FPS = 30.0
VEHICLE_SPEED_MPS = 20.0     # ground truth speed (~72 km/h / 44.7 mph)
VEHICLE_X_START_M = 12.0
VEHICLE_X_END_M = 110.0
VEHICLE_LEN_M = 4.5
VEHICLE_WID_M = 1.8
VEHICLE_LANE_Y_M = -0.9      # offset within the lane, not dead-center

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
os.makedirs(OUT_DIR, exist_ok=True)
VIDEO_PATH = os.path.join(OUT_DIR, "synthetic_scene_A.mp4")
GT_PATH = os.path.join(OUT_DIR, "synthetic_scene_A_ground_truth.json")

THETA = math.radians(PITCH_DEG)
FORWARD = np.array([math.cos(THETA), 0.0, -math.sin(THETA)])
DOWN = np.array([-math.sin(THETA), 0.0, -math.cos(THETA)])
RIGHT = np.array([0.0, 1.0, 0.0])
CAM_C = np.array([0.0, 0.0, CAM_HEIGHT_M])


def project(points_xyz):
    """World (N,3) ground points -> image (N,2) pixel coords. NaN if behind camera."""
    pts = np.atleast_2d(points_xyz).astype(float)
    d = pts - CAM_C
    zc = d @ FORWARD
    xc = d @ RIGHT
    yc = d @ DOWN
    out = np.full((len(pts), 2), np.nan)
    valid = zc > 0.5
    out[valid, 0] = CX + FOCAL_PX * xc[valid] / zc[valid]
    out[valid, 1] = CY + FOCAL_PX * yc[valid] / zc[valid]
    return out


def road_polyline(y_m, x_max=ROAD_X_MAX_M, n=400):
    xs = np.linspace(1.0, x_max, n)
    ys = np.full_like(xs, y_m)
    zs = np.zeros_like(xs)
    world = np.stack([xs, ys, zs], axis=1)
    img = project(world)
    return img


def dash_segments(x_max=ROAD_X_MAX_M):
    """Returns list of (x_start_m, x_end_m) for the dashed center line, world Y=0."""
    segs = []
    x = 0.0
    while x < x_max:
        x_end = min(x + DASH_LEN_M, x_max)
        segs.append((x, x_end))
        x += DASH_PERIOD_M
    return segs


def vehicle_corners_world(x_center_m):
    hl, hw = VEHICLE_LEN_M / 2.0, VEHICLE_WID_M / 2.0
    y = VEHICLE_LANE_Y_M
    corners = np.array([
        [x_center_m - hl, y - hw, 0.0],
        [x_center_m + hl, y - hw, 0.0],
        [x_center_m + hl, y + hw, 0.0],
        [x_center_m - hl, y + hw, 0.0],
    ])
    return corners


def draw_polyline(img, pts, color, thickness):
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 2:
        return
    pts = pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def render_background():
    img = np.full((IMG_H, IMG_W, 3), (70, 70, 72), dtype=np.uint8)  # asphalt gray
    horizon_v = project(np.array([[ROAD_X_MAX_M * 50, 0, 0]]))[0, 1]
    if not np.isnan(horizon_v):
        img[: max(0, int(horizon_v)), :] = (150, 130, 110)  # sky-ish band
    left_edge = road_polyline(-LANE_HALF_WIDTH_M)
    right_edge = road_polyline(LANE_HALF_WIDTH_M)
    draw_polyline(img, left_edge, (235, 235, 235), 3)
    draw_polyline(img, right_edge, (235, 235, 235), 3)
    for (xs, xe) in dash_segments():
        seg_world = np.array([[xs, 0.0, 0.0], [xe, 0.0, 0.0]])
        seg_img = project(seg_world)
        draw_polyline(img, seg_img, (235, 235, 235), 3)
    return img


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (IMG_W, IMG_H))

    travel = VEHICLE_X_END_M - VEHICLE_X_START_M
    duration_s = travel / VEHICLE_SPEED_MPS
    n_frames = int(duration_s * FPS)

    bg = render_background()
    rng = np.random.default_rng(7)

    gt_frames = []
    for i in range(n_frames):
        t = i / FPS
        x_c = VEHICLE_X_START_M + VEHICLE_SPEED_MPS * t
        frame = bg.copy()

        corners_w = vehicle_corners_world(x_c)
        corners_img = project(corners_w)
        if not np.isnan(corners_img).any():
            cv2.fillConvexPoly(frame, corners_img.astype(np.int32), (30, 40, 200))  # red-ish car, BGR
            cv2.polylines(frame, [corners_img.astype(np.int32)], True, (10, 10, 90), 2, cv2.LINE_AA)

        # rear ground-contact midpoint (the physically meaningful tracking point)
        rear_mid_w = np.array([[x_c - VEHICLE_LEN_M / 2.0, VEHICLE_LANE_Y_M, 0.0]])
        rear_mid_img = project(rear_mid_w)[0]

        noise = rng.normal(0, 4.0, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        writer.write(frame)
        gt_frames.append({
            "frame": i,
            "t_s": t,
            "world_x_m": x_c,
            "world_y_m": VEHICLE_LANE_Y_M,
            "speed_true_mps": VEHICLE_SPEED_MPS,
            "rear_contact_image_uv": [float(rear_mid_img[0]), float(rear_mid_img[1])],
        })
    writer.release()

    ground_truth = {
        "camera": {
            "image_w": IMG_W, "image_h": IMG_H, "focal_px": FOCAL_PX,
            "cx": CX, "cy": CY, "height_m": CAM_HEIGHT_M, "pitch_deg": PITCH_DEG,
        },
        "road": {
            "lane_half_width_m": LANE_HALF_WIDTH_M,
            "dash_len_m": DASH_LEN_M, "dash_gap_m": DASH_GAP_M,
        },
        "vehicle": {
            "speed_true_mps": VEHICLE_SPEED_MPS,
            "length_m": VEHICLE_LEN_M, "width_m": VEHICLE_WID_M,
            "lane_y_m": VEHICLE_LANE_Y_M,
        },
        "fps": FPS,
        "n_frames": n_frames,
        "frames": gt_frames,
    }
    with open(GT_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Wrote {n_frames} frames -> {VIDEO_PATH}")
    print(f"Ground truth -> {GT_PATH}")


if __name__ == "__main__":
    main()
