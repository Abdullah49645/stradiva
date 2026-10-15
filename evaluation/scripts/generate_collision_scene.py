"""
Two-vehicle collision scenario generator.

Reuses the same camera/road model as scene_generator.py (straight
single-lane road, elevated camera, dashed centerline for scale) but adds
a second vehicle so the Collision Causality & Avoidability module has
something real to operate on: a rear-end scenario where a following
vehicle (B, blue) approaches a lead vehicle (A, green) that is stopped
in the lane, brakes at a configurable time/strength, and either avoids
or collides with A depending on the parameters -- exactly the kind of
"was there a physically plausible opportunity to avoid this" question
section 15.5 (Collision Causality & Avoidability) asks about.

Ground truth includes not just positions but the *physics-derived*
avoidability verdict (computed directly from the known kinematics, not
from any CV pipeline) so the causality module's output can be checked
against an independently-computed answer.
"""
import json
import math
import os

import cv2
import numpy as np

from scene_generator import (
    IMG_W, IMG_H, LANE_HALF_WIDTH_M, FPS, OUT_DIR,
    make_camera, project, render_background, dash_segments,
)

VEHICLE_LEN_M = 4.5
VEHICLE_WID_M = 1.8
LANE_Y_M = -0.9

# physics assumptions used to compute the *ground truth* avoidability
# answer (kept identical to the assumptions the causality module itself
# will use, so this is a validation of the module's arithmetic, not a
# test of whether it guessed the right assumptions)
ASSUMED_MAX_DECEL_MPS2 = 7.0
ASSUMED_REACTION_TIME_S = 1.2


def lead_vehicle_state(t, stop_x_m, initial_v_mps=0.0, brake_start_t=0.0, brake_decel=6.0, start_x_m=None):
    """
    Lead vehicle A. Two modes:
      - initial_v_mps=0: stationary for the whole clip (original behavior;
        a genuine stress case for passive background estimation -- see
        docs/aws_and_production_notes.md).
      - initial_v_mps>0: A drives, then brakes hard starting at
        brake_start_t and comes to rest at stop_x_m -- the more common
        real rear-end scenario (a lead vehicle brake-checks or meets an
        obstacle) and, as a side effect, gives a background estimator
        multiple road-only pixel samples before A settles.
    """
    if initial_v_mps <= 0:
        return stop_x_m, 0.0
    x0 = start_x_m if start_x_m is not None else stop_x_m - required_stopping_distance(
        initial_v_mps, ASSUMED_MAX_DECEL_MPS2, 0.0)
    if t <= brake_start_t:
        return x0 + initial_v_mps * t, initial_v_mps
    dt = t - brake_start_t
    v = max(0.0, initial_v_mps - brake_decel * dt)
    t_stop = initial_v_mps / brake_decel
    if dt < t_stop:
        x = x0 + initial_v_mps * brake_start_t + initial_v_mps * dt - 0.5 * brake_decel * dt * dt
    else:
        x = x0 + initial_v_mps * brake_start_t + initial_v_mps * t_stop - 0.5 * brake_decel * t_stop * t_stop
        v = 0.0
    return x, v


def following_vehicle_state(t, v0, x0, brake_start_t, brake_decel):
    """
    Following vehicle B: constant speed v0 until brake_start_t, then
    decelerates at brake_decel (m/s^2) until stopped or (handled by
    caller) collision occurs.
    """
    if t <= brake_start_t:
        x = x0 + v0 * t
        v = v0
    else:
        dt = t - brake_start_t
        v = max(0.0, v0 - brake_decel * dt)
        t_stop = v0 / brake_decel
        if dt < t_stop:
            x = x0 + v0 * brake_start_t + v0 * dt - 0.5 * brake_decel * dt * dt
        else:
            x_at_stop = x0 + v0 * brake_start_t + v0 * t_stop - 0.5 * brake_decel * t_stop * t_stop
            x = x_at_stop
    return x, v


def required_stopping_distance(v, max_decel=ASSUMED_MAX_DECEL_MPS2, reaction_time=ASSUMED_REACTION_TIME_S):
    return v * reaction_time + (v ** 2) / (2 * max_decel)


def compute_ground_truth_avoidability(v0_at_hazard, gap_at_hazard_m):
    """
    Independent physics check: given B's speed and the gap to A at the
    moment the hazard (A stopped in-lane) became relevant, was avoidance
    physically possible under the assumed max deceleration + reaction
    time? This mirrors exactly what the causality module computes from
    reconstructed (noisy) data -- this version uses ground-truth values.
    """
    needed = required_stopping_distance(v0_at_hazard)
    return {
        "required_stopping_distance_m": needed,
        "available_gap_m": gap_at_hazard_m,
        "physically_avoidable": bool(gap_at_hazard_m >= needed),
    }


def generate_collision_scene(name, v0=25.0, brake_start_t=1.0, brake_decel=5.0,
                              b_start_x=0.0, a_stop_x=70.0, duration_s=4.5,
                              a_initial_v=0.0, a_brake_start_t=0.0, a_brake_decel=6.0):
    cam = make_camera()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(OUT_DIR, f"synthetic_collision_{name}.mp4")
    gt_path = os.path.join(OUT_DIR, f"synthetic_collision_{name}_ground_truth.json")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (IMG_W, IMG_H))

    bg = render_background(cam, include_dashes=True)
    rng = np.random.default_rng(hash(name) % (2**31))
    n_frames = int(duration_s * FPS)

    gt_frames = []
    collided = False
    collision_frame = None
    for i in range(n_frames):
        t = i / FPS
        xa, va = lead_vehicle_state(t, a_stop_x, initial_v_mps=a_initial_v,
                                     brake_start_t=a_brake_start_t, brake_decel=a_brake_decel)
        xb, vb = following_vehicle_state(t, v0, b_start_x, brake_start_t, brake_decel)

        # collision check: B's front reaches A's rear
        gap = (xa - VEHICLE_LEN_M / 2.0) - (xb + VEHICLE_LEN_M / 2.0)
        if gap <= 0 and not collided:
            collided = True
            collision_frame = i
            xb = xa - VEHICLE_LEN_M  # snap B to contact position, stop simulating further travel
            vb = 0.0

        frame = bg.copy()
        for (x_c, color) in [(xa, (60, 190, 60)), (xb, (30, 40, 200))]:  # A green, B red (BGR)
            hl, hw = VEHICLE_LEN_M / 2.0, VEHICLE_WID_M / 2.0
            corners_w = np.array([
                [x_c - hl, LANE_Y_M - hw, 0.0], [x_c + hl, LANE_Y_M - hw, 0.0],
                [x_c + hl, LANE_Y_M + hw, 0.0], [x_c - hl, LANE_Y_M + hw, 0.0],
            ])
            corners_img = project(corners_w, cam)
            if not np.isnan(corners_img).any():
                cv2.fillConvexPoly(frame, corners_img.astype(np.int32), color)
                cv2.polylines(frame, [corners_img.astype(np.int32)], True, (10, 10, 10), 2, cv2.LINE_AA)

        rear_a = project(np.array([[xa - VEHICLE_LEN_M / 2.0, LANE_Y_M, 0.0]]), cam)[0]
        rear_b = project(np.array([[xb - VEHICLE_LEN_M / 2.0, LANE_Y_M, 0.0]]), cam)[0]

        noise = rng.normal(0, 4.0, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

        gt_frames.append({
            "frame": i, "t_s": t,
            "vehicle_a": {"x_m": xa, "v_mps": va, "rear_contact_uv": [float(rear_a[0]), float(rear_a[1])]},
            "vehicle_b": {"x_m": xb, "v_mps": vb, "rear_contact_uv": [float(rear_b[0]), float(rear_b[1])]},
            "gap_m": float(gap),
        })
        if collided and i > collision_frame + 5:
            break
    writer.release()

    # ground-truth avoidability, evaluated from the earliest point in the
    # clip (t=0) -- since A is stopped in-lane for the whole clip, that's
    # the fairest "when did the hazard become observable" reference point
    # for this controlled scenario (real footage would need an actual
    # hazard-onset detector; that's out of scope here).
    v_at_hazard = v0
    gap_at_hazard = (a_stop_x - VEHICLE_LEN_M / 2.0) - (b_start_x + VEHICLE_LEN_M / 2.0)
    physics_check = compute_ground_truth_avoidability(v_at_hazard, gap_at_hazard)

    gt = {
        "scenario": name, "fps": FPS, "n_frames": len(gt_frames),
        "collided": collided, "collision_frame": collision_frame,
        "params": {"v0_mps": v0, "brake_start_t_s": brake_start_t, "brake_decel_mps2": brake_decel,
                   "b_start_x_m": b_start_x, "a_stop_x_m": a_stop_x},
        "ground_truth_physics_avoidability": physics_check,
        "frames": gt_frames,
    }
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"[{name}] {len(gt_frames)} frames, collided={collided} @frame {collision_frame} -> {video_path}")
    print(f"  physics check: {physics_check}")
    return video_path, gt_path


if __name__ == "__main__":
    # Scenario 1: brakes too late / too weakly -> physically unavoidable given assumptions
    generate_collision_scene("unavoidable", v0=25.0, brake_start_t=1.0, brake_decel=5.0,
                              a_stop_x=55.0)
    # Scenario 3: production/AWS-pipeline demo scenario -- lead vehicle A is
    # NOT parked for the whole clip (drives, then brake-checks/hits an
    # obstacle and stops), which is both more representative of real
    # rear-end collisions AND gives the production pipeline's passive
    # (no-known-camera-model) background estimator clean road samples to
    # work with, unlike the always-stationary A in the two scenarios above.
    generate_collision_scene("production_demo", v0=20.0, brake_start_t=2.2, brake_decel=5.0,
                              b_start_x=0.0, a_stop_x=75.0,
                              a_initial_v=15.0, a_brake_start_t=0.3, a_brake_decel=7.0,
                              duration_s=5.0)
    # Scenario 2: enough following distance to have stopped in time (physically
    # avoidable per the assumed physics), but braking started late/weak enough
    # that a collision still occurs -- the interesting "avoidable but wasn't
    # avoided" case the causality module needs to distinguish from scenario 1.
    # a_stop_x kept moderate (not e.g. 95m) so vehicle A stays a large enough
    # blob for the Week-1/2 placeholder color tracker to detect at all --
    # a real detector (Week 3 P0 item) won't have this constraint.
    generate_collision_scene("avoidable_but_occurred", v0=18.0, brake_start_t=2.0, brake_decel=3.5,
                              a_stop_x=60.0)
