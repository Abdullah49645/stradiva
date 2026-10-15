"""
Continuous-feed synthetic driving scene, for erratic-driving detection
(docs/erratic_driving_detection.md). Nothing like this exists yet --
every other generator in this repo is a short single-event clip
(collision or near-collision). This is a longer continuous clip of one
observed vehicle ahead of the camera, cycling through labeled segments
of normal and erratic behavior, so a detector has real ground truth to
be measured against.

Three behavior types, deliberately kept to a small, well-implemented
set rather than a large rushed one:

  NORMAL           -- near-constant speed, small realistic noise,
                       lateral position fixed at lane center.
  WEAVING           -- speed roughly constant; lateral position
                       oscillates (sinusoidal) between adjacent-lane-ish
                       extremes -- a common, safety-relevant erratic
                       pattern.
  HARSH_BRAKE_RELEASE -- speed drops sharply then partially recovers
                       (a brake pulse with no collision), lateral
                       position stays fixed -- distinct from WEAVING
                       (which is lateral, not longitudinal).

Ground truth records each segment's type, time interval, and
parameters -- this is what an evaluation script checks a detector's
flagged intervals against.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from scene_generator import IMG_W, IMG_H, VEHICLE_LEN_M, VEHICLE_WID_M, OUT_DIR, make_camera, project, render_background

LANE_Y_M = -0.9
FPS = 30.0


def _segment_kinematics(t_rel, seg):
    """Returns (x_m, y_m, speed_mps) at time t_rel (seconds since this
    segment started), given the vehicle's speed/position at the start of
    the segment (seg['x0'], seg['v0'])."""
    kind = seg["type"]
    if kind == "NORMAL":
        v = seg["v0"]
        x = seg["x0"] + v * t_rel
        y = LANE_Y_M
    elif kind == "WEAVING":
        v = seg["v0"]
        x = seg["x0"] + v * t_rel
        amplitude_m = seg.get("amplitude_m", 0.7)
        period_s = seg.get("period_s", 2.2)
        y = LANE_Y_M + amplitude_m * np.sin(2 * np.pi * t_rel / period_s)
    elif kind == "HARSH_BRAKE_RELEASE":
        v0 = seg["v0"]
        decel = seg.get("decel_mps2", 5.5)
        brake_dur = seg.get("brake_duration_s", 0.9)
        if t_rel < brake_dur:
            v = max(0.5, v0 - decel * t_rel)
            x = seg["x0"] + v0 * t_rel - 0.5 * decel * t_rel ** 2
        else:
            v_at_brake_end = max(0.5, v0 - decel * brake_dur)
            x_at_brake_end = seg["x0"] + v0 * brake_dur - 0.5 * decel * brake_dur ** 2
            reaccel = seg.get("reaccel_mps2", 2.0)
            dt2 = t_rel - brake_dur
            v = min(v0, v_at_brake_end + reaccel * dt2)
            x = x_at_brake_end + v_at_brake_end * dt2 + 0.5 * reaccel * dt2 ** 2
        y = LANE_Y_M
    else:
        raise ValueError(kind)
    return x, y, v


def generate_continuous_feed(name="feed_1", segments=None, seed=7):
    """
    segments: list of {"type": ..., "duration_s": ..., **params}. Each
    segment's starting position/speed is chained from the previous
    segment's ending state, so the whole clip is one continuous,
    physically consistent trajectory.
    """
    if segments is None:
        segments = [
            {"type": "NORMAL", "duration_s": 1.0, "v0": 7.0},
            {"type": "WEAVING", "duration_s": 1.8, "amplitude_m": 0.75, "period_s": 1.6},
            {"type": "NORMAL", "duration_s": 1.0},
            {"type": "HARSH_BRAKE_RELEASE", "duration_s": 1.8, "decel_mps2": 4.5, "brake_duration_s": 0.7},
            {"type": "NORMAL", "duration_s": 1.2},
            {"type": "WEAVING", "duration_s": 1.6, "amplitude_m": 0.5, "period_s": 1.3},
            {"type": "NORMAL", "duration_s": 1.0},
        ]

    rng = np.random.default_rng(seed)
    cam = make_camera()
    bg = render_background(cam, include_dashes=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(OUT_DIR, f"synthetic_feed_{name}.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (IMG_W, IMG_H))

    x_cursor, v_cursor, t_cursor = 5.0, 7.0, 0.0
    gt_segments = []
    frames_meta = []
    frame_idx = 0

    for seg_spec in segments:
        seg = {**seg_spec, "x0": x_cursor, "v0": seg_spec.get("v0", v_cursor)}
        duration_s = seg["duration_s"]
        n_frames = int(duration_s * FPS)
        t_start = t_cursor
        for i in range(n_frames):
            t_rel = i / FPS
            x, y, v = _segment_kinematics(t_rel, seg)
            frame = bg.copy()
            hl, hw = VEHICLE_LEN_M / 2.0, VEHICLE_WID_M / 2.0
            corners_w = np.array([
                [x - hl, y - hw, 0.0], [x + hl, y - hw, 0.0],
                [x + hl, y + hw, 0.0], [x - hl, y + hw, 0.0],
            ])
            corners_img = project(corners_w, cam)
            if not np.isnan(corners_img).any() and np.all(corners_img > -500) and np.all(corners_img < 3000):
                cv2.fillConvexPoly(frame, corners_img.astype(np.int32), (30, 40, 200))
                cv2.polylines(frame, [corners_img.astype(np.int32)], True, (10, 10, 10), 2, cv2.LINE_AA)
            noise = rng.normal(0, 4.0, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            writer.write(frame)
            frames_meta.append({"frame": frame_idx, "t_s": t_cursor + t_rel, "x_m": float(x),
                                 "y_m": float(y), "speed_mps": float(v), "segment_type": seg["type"]})
            frame_idx += 1
        x_end, y_end, v_end = _segment_kinematics(duration_s, seg)
        gt_segments.append({"type": seg["type"], "t_start_s": t_start, "t_end_s": t_start + duration_s,
                             "params": {k: v for k, v in seg.items() if k not in ("x0", "v0", "duration_s")}})
        x_cursor, v_cursor, t_cursor = x_end, v_end, t_start + duration_s

    writer.release()
    gt = {"scenario": name, "fps": FPS, "n_frames": frame_idx, "duration_s": t_cursor,
          "segments": gt_segments, "frames": frames_meta}
    gt_path = os.path.join(OUT_DIR, f"synthetic_feed_{name}_ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"[{name}] {frame_idx} frames ({t_cursor:.1f}s) across {len(gt_segments)} segments -> {video_path}")
    for s in gt_segments:
        print(f"   {s['t_start_s']:5.1f}s - {s['t_end_s']:5.1f}s : {s['type']}")
    return video_path, gt_path


if __name__ == "__main__":
    generate_continuous_feed("feed_1")
