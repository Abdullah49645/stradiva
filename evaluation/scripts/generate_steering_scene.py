"""
Steering-detection validation scenario: a following vehicle that swerves
laterally (a sharp lane change, e.g. an attempted last-second avoidance)
before reaching the lead vehicle's position, WITHOUT braking. Isolates
the steering detector from the braking detector so a true-positive check
on steering doesn't depend on braking also being present.

Ground truth for the swerve is known exactly (a smoothstep lateral
offset over a fixed time window), independent of the CV pipeline,
matching this project's validate-against-independent-ground-truth
discipline used throughout (see week1-4 assessments).
"""
import json
import os

import cv2
import numpy as np

from scene_generator import (
    IMG_W, IMG_H, FPS, OUT_DIR, make_camera, project, render_background,
)

VEHICLE_LEN_M = 4.5
VEHICLE_WID_M = 1.8
LANE_Y_START_M = -0.9
LANE_Y_END_M = 1.5      # swerves across to roughly the other side of the lane
SWERVE_START_T = 0.8
SWERVE_DURATION_S = 0.6
VEHICLE_SPEED_MPS = 20.0
VEHICLE_X_START_M = 10.0
DURATION_S = 3.0


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def lateral_offset(t):
    frac = (t - SWERVE_START_T) / SWERVE_DURATION_S
    return LANE_Y_START_M + (LANE_Y_END_M - LANE_Y_START_M) * smoothstep(frac)


def generate_swerve_scene(name="steering_validation"):
    cam = make_camera()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(OUT_DIR, f"synthetic_{name}.mp4")
    gt_path = os.path.join(OUT_DIR, f"synthetic_{name}_ground_truth.json")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (IMG_W, IMG_H))

    bg = render_background(cam, include_dashes=True)
    rng = np.random.default_rng(11)
    n_frames = int(DURATION_S * FPS)

    gt_frames = []
    for i in range(n_frames):
        t = i / FPS
        x_c = VEHICLE_X_START_M + VEHICLE_SPEED_MPS * t
        y_c = lateral_offset(t)

        frame = bg.copy()
        hl, hw = VEHICLE_LEN_M / 2.0, VEHICLE_WID_M / 2.0
        corners_w = np.array([
            [x_c - hl, y_c - hw, 0.0], [x_c + hl, y_c - hw, 0.0],
            [x_c + hl, y_c + hw, 0.0], [x_c - hl, y_c + hw, 0.0],
        ])
        corners_img = project(corners_w, cam)
        if not np.isnan(corners_img).any():
            cv2.fillConvexPoly(frame, corners_img.astype(np.int32), (30, 40, 200))
            cv2.polylines(frame, [corners_img.astype(np.int32)], True, (10, 10, 10), 2, cv2.LINE_AA)
        rear = project(np.array([[x_c - hl, y_c, 0.0]]), cam)[0]

        noise = rng.normal(0, 4.0, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

        gt_frames.append({"frame": i, "t_s": t, "world_x_m": x_c, "world_y_m": y_c,
                           "rear_contact_uv": [float(rear[0]), float(rear[1])]})
    writer.release()

    gt = {
        "scenario": name, "fps": FPS, "n_frames": n_frames,
        "swerve_start_t_s": SWERVE_START_T, "swerve_duration_s": SWERVE_DURATION_S,
        "lane_y_start_m": LANE_Y_START_M, "lane_y_end_m": LANE_Y_END_M,
        "vehicle_speed_mps": VEHICLE_SPEED_MPS,
        "frames": gt_frames,
    }
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"[{name}] {n_frames} frames -> {video_path}")
    return video_path, gt_path


if __name__ == "__main__":
    generate_swerve_scene()
