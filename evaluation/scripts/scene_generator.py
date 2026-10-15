"""
Parameterized version of generate_synthetic_scene.py: same camera model
and road/vehicle setup, but exposes camera roll, dash-marking presence,
and post-render degradation (fog/blur) as knobs, so Week 2 can generate
a small deliberate-failure dataset per brief section 26 ("Failure
Dataset") without duplicating the whole generator per scenario.

Camera model matches generate_synthetic_scene.py exactly, with one
addition: an optional roll rotation (about the camera's forward axis),
since real footage is essentially never perfectly level and the Week 1
pipeline assumes zero roll throughout.
"""
import json
import math
import os
import zlib

import cv2
import numpy as np

IMG_W, IMG_H = 1280, 720
FOCAL_PX = 950.0
CX, CY = IMG_W / 2.0, IMG_H / 2.0
CAM_HEIGHT_M = 6.0
PITCH_DEG = 22.0
LANE_HALF_WIDTH_M = 1.85
DASH_LEN_M = 3.0
DASH_GAP_M = 9.0
DASH_PERIOD_M = DASH_LEN_M + DASH_GAP_M
ROAD_X_MAX_M = 160.0

FPS = 30.0
VEHICLE_SPEED_MPS = 20.0
VEHICLE_X_START_M = 12.0
VEHICLE_X_END_M = 110.0
VEHICLE_LEN_M = 4.5
VEHICLE_WID_M = 1.8
VEHICLE_LANE_Y_M = -0.9

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
os.makedirs(OUT_DIR, exist_ok=True)


def make_camera(pitch_deg=PITCH_DEG, roll_deg=0.0, height_m=CAM_HEIGHT_M,
                 yaw_deg=0.0, x_offset_m=0.0, y_offset_m=0.0):
    """
    yaw_deg/x_offset_m/y_offset_m default to 0.0, reproducing the
    original single-camera-at-the-road-origin-looking-straight-down-it
    behavior exactly -- added for the multi-camera work (see
    docs/multi_camera_architecture.md) where a second "witness" camera
    needs to sit off to the side of the road at its own position/angle,
    not just vary pitch/roll/height along the same viewing axis.
    """
    theta = math.radians(pitch_deg)
    psi = math.radians(yaw_deg)
    z_hat = np.array([0.0, 0.0, 1.0])
    hf = np.array([math.cos(psi), math.sin(psi), 0.0])   # horizontal forward
    hr = np.array([-math.sin(psi), math.cos(psi), 0.0])  # horizontal right (hr(0)=[0,1,0], matches the original)
    forward0 = math.cos(theta) * hf - math.sin(theta) * z_hat
    down0 = -math.sin(theta) * hf - math.cos(theta) * z_hat
    right0 = hr
    if abs(roll_deg) > 1e-9:
        phi = math.radians(roll_deg)
        c, s = math.cos(phi), math.sin(phi)
        right = c * right0 + s * down0
        down = -s * right0 + c * down0
        forward = forward0
    else:
        right, down, forward = right0, down0, forward0
    cam_c = np.array([x_offset_m, y_offset_m, height_m])
    return {"forward": forward, "down": down, "right": right, "c": cam_c}


def project(points_xyz, cam, focal_px=FOCAL_PX):
    pts = np.atleast_2d(points_xyz).astype(float)
    d = pts - cam["c"]
    zc = d @ cam["forward"]
    xc = d @ cam["right"]
    yc = d @ cam["down"]
    out = np.full((len(pts), 2), np.nan)
    valid = zc > 0.5
    out[valid, 0] = CX + focal_px * xc[valid] / zc[valid]
    out[valid, 1] = CY + focal_px * yc[valid] / zc[valid]
    return out


def road_polyline(y_m, cam, x_max=ROAD_X_MAX_M, n=400):
    xs = np.linspace(1.0, x_max, n)
    world = np.stack([xs, np.full_like(xs, y_m), np.zeros_like(xs)], axis=1)
    return project(world, cam)


def dash_segments(x_max=ROAD_X_MAX_M):
    segs, x = [], 0.0
    while x < x_max:
        segs.append((x, min(x + DASH_LEN_M, x_max)))
        x += DASH_PERIOD_M
    return segs


def vehicle_corners_world(x_center_m):
    hl, hw = VEHICLE_LEN_M / 2.0, VEHICLE_WID_M / 2.0
    y = VEHICLE_LANE_Y_M
    return np.array([
        [x_center_m - hl, y - hw, 0.0], [x_center_m + hl, y - hw, 0.0],
        [x_center_m + hl, y + hw, 0.0], [x_center_m - hl, y + hw, 0.0],
    ])


def draw_polyline(img, pts, color, thickness):
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 2:
        return
    cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def render_background(cam, include_dashes=True):
    img = np.full((IMG_H, IMG_W, 3), (70, 70, 72), dtype=np.uint8)
    horizon_v = project(np.array([[ROAD_X_MAX_M * 50, 0, 0]]), cam)[0, 1]
    if not np.isnan(horizon_v):
        img[: max(0, int(horizon_v)), :] = (150, 130, 110)
    draw_polyline(img, road_polyline(-LANE_HALF_WIDTH_M, cam), (235, 235, 235), 3)
    draw_polyline(img, road_polyline(LANE_HALF_WIDTH_M, cam), (235, 235, 235), 3)
    if include_dashes:
        for xs, xe in dash_segments():
            seg = project(np.array([[xs, 0.0, 0.0], [xe, 0.0, 0.0]]), cam)
            draw_polyline(img, seg, (235, 235, 235), 3)
    return img


def apply_fog(frame, strength=0.85, blur_ksize=55):
    fog = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)
    return cv2.addWeighted(fog, 1 - strength, np.full_like(fog, 120), strength, 0)


def generate_scene(name, roll_deg=0.0, include_dashes=True, fog=False,
                    speed_mps=VEHICLE_SPEED_MPS):
    cam = make_camera(roll_deg=roll_deg)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(OUT_DIR, f"synthetic_scene_{name}.mp4")
    gt_path = os.path.join(OUT_DIR, f"synthetic_scene_{name}_ground_truth.json")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (IMG_W, IMG_H))

    travel = VEHICLE_X_END_M - VEHICLE_X_START_M
    n_frames = int((travel / speed_mps) * FPS)
    bg = render_background(cam, include_dashes=include_dashes)
    # NOTE: this used to be `hash(name) % (2**31)`. Python's built-in
    # hash() for strings is salted with a per-process random seed
    # (PYTHONHASHSEED, on by default since Python 3.3, a security
    # feature unrelated to this use case) -- so the "same name -> same
    # scene" reproducibility this function's callers rely on (stress
    # tests, calibration sweeps, demo data) was silently false: every
    # fresh process produced a genuinely different random scene for an
    # identical name. zlib.crc32 has no such per-process salt.
    rng = np.random.default_rng(zlib.crc32(name.encode()) % (2**31))

    gt_frames = []
    for i in range(n_frames):
        t = i / FPS
        x_c = VEHICLE_X_START_M + speed_mps * t
        frame = bg.copy()
        corners_img = project(vehicle_corners_world(x_c), cam)
        if not np.isnan(corners_img).any():
            cv2.fillConvexPoly(frame, corners_img.astype(np.int32), (30, 40, 200))
            cv2.polylines(frame, [corners_img.astype(np.int32)], True, (10, 10, 90), 2, cv2.LINE_AA)
        rear_mid = project(np.array([[x_c - VEHICLE_LEN_M / 2.0, VEHICLE_LANE_Y_M, 0.0]]), cam)[0]

        noise = rng.normal(0, 4.0, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if fog:
            frame = apply_fog(frame)

        writer.write(frame)
        gt_frames.append({
            "frame": i, "t_s": t, "world_x_m": x_c, "world_y_m": VEHICLE_LANE_Y_M,
            "speed_true_mps": speed_mps,
            "rear_contact_image_uv": [float(rear_mid[0]), float(rear_mid[1])],
        })
    writer.release()

    gt = {
        "scenario": name,
        "camera": {"image_w": IMG_W, "image_h": IMG_H, "focal_px": FOCAL_PX,
                    "cx": CX, "cy": CY, "height_m": CAM_HEIGHT_M,
                    "pitch_deg": PITCH_DEG, "roll_deg": roll_deg},
        "road": {"lane_half_width_m": LANE_HALF_WIDTH_M, "include_dashes": include_dashes,
                 "dash_len_m": DASH_LEN_M, "dash_gap_m": DASH_GAP_M, "fog": fog},
        "vehicle": {"speed_true_mps": speed_mps, "length_m": VEHICLE_LEN_M,
                    "width_m": VEHICLE_WID_M, "lane_y_m": VEHICLE_LANE_Y_M},
        "fps": FPS, "n_frames": n_frames, "frames": gt_frames,
    }
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"[{name}] {n_frames} frames -> {video_path}")
    return video_path, gt_path


if __name__ == "__main__":
    generate_scene("A")  # baseline, matches original generate_synthetic_scene.py output
