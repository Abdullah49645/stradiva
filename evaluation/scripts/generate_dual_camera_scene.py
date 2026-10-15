"""
Two-camera synthetic collision generator, for the multi-camera
cross-examination work (docs/multi_camera_architecture.md).

No real unsynchronized multi-view crash footage exists to build against
-- this generates a controlled substitute: the SAME physical collision
event (identical vehicle kinematics, reused directly from
generate_collision_scene.py) rendered from two independently-positioned
cameras, at two different frame rates, with a deliberate, hidden time
offset between the two recordings' start times. This mirrors the real
case this feature targets: a following vehicle's dashcam and a business's
CCTV camera further down the same street, neither aware of the other,
neither frame-synchronized.

Camera A: identical to generate_collision_scene.py's camera (elevated,
at the road's coordinate origin, looking down +x).
Camera B: positioned at the OPPOSITE end of the same straight road,
facing back toward camera A (yaw=180 deg) -- a different, independently-
computed viewing geometry of the same event, not just a translated copy
of camera A's view.

Ground truth records the true time offset and each camera's true
extrinsics FOR VALIDATION ONLY. The reconstruction pipeline must recover
each camera's geometry independently (exactly as it does for a single
camera) and estimate the time offset from the footage itself -- neither
is ever handed to it.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from scene_generator import (
    IMG_W, IMG_H, VEHICLE_LEN_M, OUT_DIR, make_camera, project, render_background,
)
from generate_collision_scene import (
    lead_vehicle_state, following_vehicle_state, compute_ground_truth_avoidability,
    LANE_Y_M,
)

CAM_B_YAW_DEG = 180.0
CAM_B_HEIGHT_M = 8.0
CAM_B_PITCH_DEG = 18.0


def _render_camera(name, cam, fps, duration_s, time_offset_s, scenario_kwargs, rng_seed_extra):
    """Renders one camera's video of the scenario, starting its own
    clock at time_offset_s relative to the shared t=0 of the physical
    event (t=0 is when generate_collision_scene's kinematics start, the
    same instant for both cameras -- time_offset_s only changes when
    EACH CAMERA'S RECORDING starts relative to that shared instant,
    exactly like two independent people hitting record at different
    real-world moments)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(OUT_DIR, f"synthetic_dualcam_{name}.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (IMG_W, IMG_H))
    bg = render_background(cam, include_dashes=True)
    rng = np.random.default_rng((abs(hash(name)) + rng_seed_extra) % (2**31))
    n_frames = int(duration_s * fps)

    frames_meta = []
    for i in range(n_frames):
        t_local = i / fps
        t_shared = t_local + time_offset_s  # this camera's frame i corresponds to shared-clock time t_shared
        xa, va = lead_vehicle_state(t_shared, scenario_kwargs["a_stop_x"],
                                     initial_v_mps=scenario_kwargs.get("a_initial_v", 0.0),
                                     brake_start_t=scenario_kwargs.get("a_brake_start_t", 0.0),
                                     brake_decel=scenario_kwargs.get("a_brake_decel", 6.0))
        xb, vb = following_vehicle_state(t_shared, scenario_kwargs["v0"], scenario_kwargs["b_start_x"],
                                          scenario_kwargs["brake_start_t"], scenario_kwargs["brake_decel"])
        gap = (xa - VEHICLE_LEN_M / 2.0) - (xb + VEHICLE_LEN_M / 2.0)
        if gap <= 0:
            xb = xa - VEHICLE_LEN_M
            vb = 0.0

        frame = bg.copy()
        for (x_c, color) in [(xa, (60, 190, 60)), (xb, (30, 40, 200))]:
            hl, hw = VEHICLE_LEN_M / 2.0, 1.8 / 2.0
            corners_w = np.array([
                [x_c - hl, LANE_Y_M - hw, 0.0], [x_c + hl, LANE_Y_M - hw, 0.0],
                [x_c + hl, LANE_Y_M + hw, 0.0], [x_c - hl, LANE_Y_M + hw, 0.0],
            ])
            corners_img = project(corners_w, cam)
            if not np.isnan(corners_img).any() and np.all(corners_img > -500) and np.all(corners_img < 3000):
                cv2.fillConvexPoly(frame, corners_img.astype(np.int32), color)
                cv2.polylines(frame, [corners_img.astype(np.int32)], True, (10, 10, 10), 2, cv2.LINE_AA)

        noise = rng.normal(0, 4.0, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
        frames_meta.append({"frame": i, "t_local_s": t_local, "t_shared_s": t_shared,
                             "vehicle_a_x_m": xa, "vehicle_b_x_m": xb, "gap_m": float(gap)})
    writer.release()
    return video_path, frames_meta


def generate_dual_camera_scene(name="collision_1", v0=22.0, brake_start_t=1.2, brake_decel=4.5,
                                b_start_x=0.0, a_stop_x=65.0, duration_s=5.0,
                                fps_a=30.0, fps_b=24.0, cam_a_offset_s=0.0, cam_b_offset_s=0.4):
    """
    cam_a_offset_s / cam_b_offset_s: when each camera's recording starts,
    relative to the shared physical-event clock (t=0 = when the
    kinematics in generate_collision_scene.py's model start). The
    DIFFERENCE between these (cam_b_offset_s - cam_a_offset_s) is the
    true, hidden time offset a sync-estimation algorithm must recover
    from the footage alone.
    """
    scenario_kwargs = {"v0": v0, "brake_start_t": brake_start_t, "brake_decel": brake_decel,
                        "b_start_x": b_start_x, "a_stop_x": a_stop_x}

    cam_a = make_camera()  # identical to the single-camera generator's default
    cam_b = make_camera(pitch_deg=CAM_B_PITCH_DEG, yaw_deg=CAM_B_YAW_DEG, height_m=CAM_B_HEIGHT_M,
                         x_offset_m=a_stop_x + 35.0, y_offset_m=0.0)

    video_a, meta_a = _render_camera(f"{name}_A", cam_a, fps_a, duration_s, cam_a_offset_s, scenario_kwargs, 1)
    video_b, meta_b = _render_camera(f"{name}_B", cam_b, fps_b, duration_s, cam_b_offset_s, scenario_kwargs, 2)

    v_at_hazard = v0
    gap_at_hazard = (a_stop_x - VEHICLE_LEN_M / 2.0) - (b_start_x + VEHICLE_LEN_M / 2.0)
    physics_check = compute_ground_truth_avoidability(v_at_hazard, gap_at_hazard)

    gt = {
        "scenario": name,
        "true_time_offset_s_B_minus_A": cam_b_offset_s - cam_a_offset_s,
        "camera_a": {"video": video_a, "fps": fps_a, "start_offset_s": cam_a_offset_s,
                     "true_extrinsics": {"pitch_deg": 22.0, "yaw_deg": 0.0, "height_m": 6.0, "x_offset_m": 0.0, "y_offset_m": 0.0}},
        "camera_b": {"video": video_b, "fps": fps_b, "start_offset_s": cam_b_offset_s,
                     "true_extrinsics": {"pitch_deg": CAM_B_PITCH_DEG, "yaw_deg": CAM_B_YAW_DEG,
                                         "height_m": CAM_B_HEIGHT_M, "x_offset_m": a_stop_x + 35.0, "y_offset_m": 0.0}},
        "params": scenario_kwargs,
        "ground_truth_physics_avoidability": physics_check,
        "frames_a": meta_a, "frames_b": meta_b,
    }
    gt_path = os.path.join(OUT_DIR, f"synthetic_dualcam_{name}_ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"[{name}] camera A: {len(meta_a)} frames @ {fps_a}fps (starts at shared t={cam_a_offset_s}s) -> {video_a}")
    print(f"[{name}] camera B: {len(meta_b)} frames @ {fps_b}fps (starts at shared t={cam_b_offset_s}s) -> {video_b}")
    print(f"[{name}] true hidden offset (B - A): {cam_b_offset_s - cam_a_offset_s:+.2f}s")
    return video_a, video_b, gt_path


if __name__ == "__main__":
    generate_dual_camera_scene("collision_1")
