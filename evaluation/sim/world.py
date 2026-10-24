"""
Stradiva scene simulator v2 -- the instrument the evaluation is built on.

Why a second simulator (evaluation/scripts/scene_generator.py still exists
and still reproduces the original fixtures): the original generator draws
every scene from *exactly* the road dimensions the pipeline assumes
(3.7 m between edge lines, 3 m dashes, 9 m gaps), renders vehicles as flat
coloured polygons painted onto the road, and always starts the dash
pattern at the same phase. Each of those choices makes the reconstruction
look better than it can be on a real road:

* exact priors -> the dominant real-world error source (the true marking
  dimensions differ from the assumed standard) is zero by construction;
* flat vehicles -> the blob's bottom edge is exactly the ground contact,
  and a vehicle never occludes the markings or another vehicle;
* fixed dash phase -> the "nearest visible dash" is always a whole dash.

This module keeps the same pinhole camera convention as scene_generator.py
(verified to produce identical projections at the old defaults, see
tests/test_sim_world.py) and adds, all as explicit, recorded parameters:

* true road dimensions that may differ from what the pipeline assumes;
* random dash phase, worn/missing paint, low-contrast paint;
* 3D box vehicles (painter-ordered faces + contact shadow) with scripted
  longitudinal/lateral motion (cruise, brake, stop, swerve);
* a *moving* camera option (forward dashcam) for the ego-speed path;
* camera roll / yaw / pitch / height / focal length, per-frame shake;
* depth-dependent fog, blur, sensor noise, brightness change.

Everything the renderer knows is written to the ground-truth JSON. The
pipeline never imports this module.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------

@dataclass
class CameraSpec:
    image_w: int = 1280
    image_h: int = 720
    focal_px: float = 950.0
    height_m: float = 6.0
    pitch_deg: float = 22.0
    roll_deg: float = 0.0
    yaw_deg: float = 0.0
    x_m: float = 0.0          # camera position along the road (moving camera: initial value)
    y_m: float = 0.0          # lateral position
    speed_mps: float = 0.0    # >0 => forward-moving (dashcam) camera
    shake_deg: float = 0.0    # per-frame random roll/pitch jitter (std, degrees)
    accel_segments: list = None   # optional [(t_start, accel_mps2), ...] for a moving camera
    hood_frac: float = 0.0        # dark bonnet occupying the bottom fraction of the image (dashcam)


def camera_axes(pitch_deg, roll_deg, yaw_deg):
    """Same convention as scene_generator.make_camera (world X forward along
    the road, Y lateral, Z up; image right/down axes)."""
    theta, psi = math.radians(pitch_deg), math.radians(yaw_deg)
    z_hat = np.array([0.0, 0.0, 1.0])
    hf = np.array([math.cos(psi), math.sin(psi), 0.0])
    hr = np.array([-math.sin(psi), math.cos(psi), 0.0])
    forward = math.cos(theta) * hf - math.sin(theta) * z_hat
    down0 = -math.sin(theta) * hf - math.cos(theta) * z_hat
    right0 = hr
    phi = math.radians(roll_deg)
    c, s = math.cos(phi), math.sin(phi)
    right = c * right0 + s * down0
    down = -s * right0 + c * down0
    return forward, right, down


class Camera:
    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self.set_pose(spec.x_m, spec.pitch_deg, spec.roll_deg, spec.yaw_deg)

    def set_pose(self, x_m, pitch_deg, roll_deg, yaw_deg):
        s = self.spec
        self.forward, self.right, self.down = camera_axes(pitch_deg, roll_deg, yaw_deg)
        self.c = np.array([x_m, s.y_m, s.height_m])

    def project(self, pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        d = pts - self.c
        zc = d @ self.forward
        xc = d @ self.right
        yc = d @ self.down
        out = np.full((len(pts), 2), np.nan)
        ok = zc > 0.3
        f = self.spec.focal_px
        out[ok, 0] = self.spec.image_w / 2.0 + f * xc[ok] / zc[ok]
        out[ok, 1] = self.spec.image_h / 2.0 + f * yc[ok] / zc[ok]
        return out, zc

    def ground_homography_world_to_image(self):
        """Exact 3x3 homography mapping road-plane (X, Y) -> image (u, v).
        Used only by tests/ground truth, never by the pipeline."""
        f, cx, cy = self.spec.focal_px, self.spec.image_w / 2.0, self.spec.image_h / 2.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        R = np.stack([self.right, self.down, self.forward])  # rows: camera axes in world
        t = -R @ self.c
        return K @ np.column_stack([R[:, 0], R[:, 1], t])


# --------------------------------------------------------------------------
# Road
# --------------------------------------------------------------------------

@dataclass
class RoadSpec:
    # "centerline": solid edges at +-edge_half_width, one dashed line midway
    #               (the topology the original pipeline was built for).
    # "multilane":  n_lanes lanes of lane_width, solid outer edges, dashed
    #               separators (real freeway topology; used by the dashcam path).
    topology: str = "centerline"
    edge_half_width_m: float = 1.85
    lane_width_m: float = 3.66
    n_lanes: int = 3
    dash_len_m: float = 3.0
    dash_gap_m: float = 9.0
    dash_phase_m: float = 0.0
    line_width_m: float = 0.15
    paint_level: int = 235       # paint grey level (low contrast -> closer to asphalt)
    asphalt_level: int = 72
    wear_prob: float = 0.0       # probability a dash is missing entirely
    x_max_m: float = 220.0
    x_min_m: float = -30.0


def _dash_intervals(spec: RoadSpec, x_lo, x_hi, rng):
    period = spec.dash_len_m + spec.dash_gap_m
    k0 = math.floor((x_lo - spec.dash_phase_m) / period) - 1
    out = []
    k = k0
    while True:
        xs = spec.dash_phase_m + k * period
        if xs > x_hi:
            break
        keep = True
        if spec.wear_prob > 0:
            # deterministic per-dash wear so a moving camera sees the same road
            keep = (np.random.default_rng(abs(hash((k, 7919))) % (2**32)).random() >= spec.wear_prob)
        if keep and xs + spec.dash_len_m > x_lo:
            out.append((xs, xs + spec.dash_len_m))
        k += 1
    return out


def road_lines(spec: RoadSpec):
    """Returns list of (y_m, kind) with kind in {'solid','dashed'}."""
    if spec.topology == "centerline":
        return [(-spec.edge_half_width_m, "solid"), (0.0, "dashed"), (spec.edge_half_width_m, "solid")]
    w = spec.lane_width_m
    n = spec.n_lanes
    ys = [(-n / 2.0 + i) * w for i in range(n + 1)]
    return [(y, "solid" if i in (0, n) else "dashed") for i, y in enumerate(ys)]


def _fill_world_quad(img, cam: Camera, corners, color, near_x=None):
    pts, zc = cam.project(corners)
    if np.isnan(pts).any():
        return
    cv2.fillConvexPoly(img, np.round(pts * 16).astype(np.int32), color, cv2.LINE_AA, shift=4)


def render_road(cam: Camera, spec: RoadSpec, rng, depth_buffer=None):
    s = cam.spec
    img = np.full((s.image_h, s.image_w, 3), spec.asphalt_level, dtype=np.uint8)
    img[..., 2] = np.clip(spec.asphalt_level + 2, 0, 255)
    # sky above the horizon
    horizon, _ = cam.project(np.array([[cam.c[0] + 5000.0 * cam.forward[0], cam.c[1] + 5000.0 * cam.forward[1], 0.0]]))
    hv = horizon[0, 1]
    if not np.isnan(hv):
        img[: int(np.clip(hv, 0, s.image_h)), :] = (150, 130, 110)
    x_lo = max(spec.x_min_m, cam.c[0] + 0.5)
    x_hi = min(spec.x_max_m + cam.c[0], cam.c[0] + 400.0)
    hw = spec.line_width_m / 2.0
    paint = (spec.paint_level,) * 3
    for y, kind in road_lines(spec):
        if kind == "solid":
            # a straight 3D line projects to a straight image line, so one quad suffices
            _fill_world_quad(img, cam, np.array([[x_lo, y - hw, 0], [x_hi, y - hw, 0], [x_hi, y + hw, 0], [x_lo, y + hw, 0]]), paint)
        else:
            for a, b in _dash_intervals(spec, x_lo, x_hi, rng):
                a, b = max(a, x_lo), min(b, x_hi)
                if b <= a:
                    continue
                _fill_world_quad(img, cam, np.array([[a, y - hw, 0], [b, y - hw, 0], [b, y + hw, 0], [a, y + hw, 0]]), paint)
    return img


# --------------------------------------------------------------------------
# Vehicles
# --------------------------------------------------------------------------

@dataclass
class MotionSegment:
    """Constant longitudinal acceleration from t_start until the next segment.
    Speed is clamped at >= 0 (a stopped vehicle stays stopped)."""
    t_start_s: float
    accel_mps2: float


@dataclass
class VehicleSpec:
    vid: str
    x0_m: float
    v0_mps: float
    lane_y_m: float
    length_m: float = 4.5
    width_m: float = 1.8
    height_m: float = 1.5
    color_bgr: tuple = (30, 40, 200)
    segments: List[MotionSegment] = field(default_factory=list)
    direction: int = 1        # +1 travels away from the camera (+X), -1 towards it
    # lateral manoeuvre: (t_start, duration, dy) cosine-ramp lane change
    swerve: Optional[tuple] = None


def simulate_vehicle(spec: VehicleSpec, t):
    """Integrates the scripted motion exactly (piecewise-constant accel,
    speed floored at zero). Returns x, v, a arrays at times t."""
    segs = sorted(spec.segments, key=lambda s: s.t_start_s)
    dt = 1e-3
    tt = np.arange(0.0, float(np.max(t)) + 2 * dt, dt)
    acc = np.zeros_like(tt)
    for i, s in enumerate(segs):
        t_end = segs[i + 1].t_start_s if i + 1 < len(segs) else np.inf
        acc[(tt >= s.t_start_s) & (tt < t_end)] = s.accel_mps2
    v = np.empty_like(tt)
    x = np.empty_like(tt)
    v[0], x[0] = spec.v0_mps, spec.x0_m
    for i in range(1, len(tt)):
        v[i] = max(0.0, v[i - 1] + acc[i - 1] * dt)
        x[i] = x[i - 1] + 0.5 * (v[i] + v[i - 1]) * dt
    a_eff = np.gradient(v, tt)
    d = spec.direction
    x_signed = spec.x0_m + d * (x - spec.x0_m)
    return np.interp(t, tt, x_signed), np.interp(t, tt, v), np.interp(t, tt, a_eff)


def lateral_offset(spec: VehicleSpec, t):
    if not spec.swerve:
        return np.full_like(np.asarray(t, dtype=float), spec.lane_y_m)
    t0, dur, dy = spec.swerve
    frac = np.clip((np.asarray(t) - t0) / dur, 0, 1)
    return spec.lane_y_m + dy * (1 - np.cos(np.pi * frac)) / 2.0


def box_corners(x, y, L, W, H):
    hl, hw = L / 2.0, W / 2.0
    base = np.array([[x - hl, y - hw], [x + hl, y - hw], [x + hl, y + hw], [x - hl, y + hw]])
    bottom = np.column_stack([base, np.zeros(4)])
    top = np.column_stack([base, np.full(4, H)])
    return np.vstack([bottom, top])


_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
_FACE_SHADE = [0.35, 1.0, 0.75, 0.85, 0.6, 0.7]


def draw_vehicle(img, cam: Camera, x, y, vs: VehicleSpec):
    # contact shadow (slightly larger than the footprint)
    sh = box_corners(x, y, vs.length_m + 0.3, vs.width_m + 0.3, 0.0)[:4]
    pts, _ = cam.project(sh)
    if not np.isnan(pts).any():
        x0, y0 = np.maximum(np.floor(pts.min(axis=0)).astype(int) - 2, 0)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 2
        x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1]
            overlay = roi.copy()
            cv2.fillConvexPoly(overlay, np.round((pts - [x0, y0]) * 16).astype(np.int32), (25, 25, 25), cv2.LINE_AA, shift=4)
            cv2.addWeighted(overlay, 0.6, roi, 0.4, 0, dst=roi)
    corners = box_corners(x, y, vs.length_m, vs.width_m, vs.height_m)
    pts, zc = cam.project(corners)
    if np.isnan(pts).any():
        return None
    centre = corners.mean(axis=0)
    faces = []
    for fi, f in enumerate(_FACES):
        fc = corners[list(f)]
        n = np.cross(fc[1] - fc[0], fc[2] - fc[0])
        if np.dot(n, fc.mean(axis=0) - centre) < 0:
            n = -n
        if np.dot(n, cam.c - fc.mean(axis=0)) <= 0:
            continue  # back face
        depth = float(np.mean(zc[list(f)]))
        faces.append((depth, fi, f))
    for depth, fi, f in sorted(faces, reverse=True):
        col = tuple(int(np.clip(c * _FACE_SHADE[fi], 0, 255)) for c in vs.color_bgr)
        cv2.fillConvexPoly(img, np.round(pts[list(f)] * 16).astype(np.int32), col, cv2.LINE_AA, shift=4)
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


# --------------------------------------------------------------------------
# Degradations
# --------------------------------------------------------------------------

@dataclass
class Degradation:
    noise_std: float = 4.0
    blur_px: float = 0.0          # Gaussian blur sigma
    fog_visibility_m: float = 0.0  # 0 = off; else exp(-depth/visibility) contrast loss
    brightness: float = 1.0        # global gain (night ~0.35)


def apply_degradation(img, deg: Degradation, rng, depth_map=None):
    out = img.astype(np.float32)
    if deg.fog_visibility_m > 0 and depth_map is not None:
        t = np.exp(-depth_map / deg.fog_visibility_m)[..., None]
        out = out * t + 140.0 * (1 - t)
    out *= deg.brightness
    if deg.blur_px > 0:
        out = cv2.GaussianBlur(out, (0, 0), deg.blur_px)
    if deg.noise_std > 0:
        out += rng.normal(0, deg.noise_std, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def ground_depth_map(cam: Camera):
    """Per-pixel distance along the camera forward axis to the road plane
    (inf above the horizon) -- used only for fog rendering."""
    s = cam.spec
    u, v = np.meshgrid(np.arange(s.image_w), np.arange(s.image_h))
    rays = ((u - s.image_w / 2.0)[..., None] * cam.right + (v - s.image_h / 2.0)[..., None] * cam.down
            + s.focal_px * cam.forward) / s.focal_px
    rz = rays[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -cam.c[2] / rz
    lam[(rz >= 0) | ~np.isfinite(lam)] = 1e4
    return np.minimum(lam, 1e4)


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    seed: int = 0
    fps: float = 30.0
    duration_s: float = 5.0
    camera: CameraSpec = field(default_factory=CameraSpec)
    road: RoadSpec = field(default_factory=RoadSpec)
    vehicles: List[VehicleSpec] = field(default_factory=list)
    degradation: Degradation = field(default_factory=Degradation)
    post_collision_decel_mps2: float = 6.0
    notes: str = ""


def _collision_time(t, xs, vehicles):
    """First time a follower's front reaches its lead's rear (same lane, same
    travel direction, follower behind in the direction of travel at t=0).
    Returns (t_contact, follower_id, lead_id) or None."""
    best = None
    for a in vehicles:
        for b in vehicles:
            if a is b or a.direction != b.direction:
                continue
            d = a.direction
            ya, yb = lateral_offset(a, t), lateral_offset(b, t)
            same_lane = np.abs(ya - yb) < (a.width_m + b.width_m) / 2.0
            gap = d * (xs[b.vid] - xs[a.vid]) - (a.length_m + b.length_m) / 2.0
            if not d * (xs[b.vid][0] - xs[a.vid][0]) > 0:
                continue
            hit = np.where(same_lane & (gap <= 0))[0]
            if len(hit):
                k = hit[0]
                if k > 0:
                    g0, g1 = gap[k - 1], gap[k]
                    tc = t[k - 1] + (t[k] - t[k - 1]) * g0 / (g0 - g1) if g0 != g1 else t[k]
                else:
                    tc = t[k]
                if best is None or tc < best[0]:
                    best = (float(tc), a.vid, b.vid)
    return best


def render_scenario(sc: Scenario, out_dir: str, write_video=True):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(sc.seed)
    n = int(round(sc.duration_s * sc.fps))
    t = np.arange(n) / sc.fps
    cam = Camera(sc.camera)
    s = sc.camera

    # free (uncollided) scripted motion
    free = {v.vid: simulate_vehicle(v, t) for v in sc.vehicles}
    xs = {k: val[0].copy() for k, val in free.items()}
    vs = {k: val[1].copy() for k, val in free.items()}
    col = _collision_time(t, xs, sc.vehicles)
    if col is not None:
        tc, fid, lid = col
        # after contact the pair moves together and decelerates to rest
        k0 = int(np.searchsorted(t, tc))
        if k0 < n:
            vf = {v.vid: v for v in sc.vehicles}
            d = vf[lid].direction
            v_common = 0.5 * (vs[fid][k0] + vs[lid][k0])
            x_lead = xs[lid][k0 - 1] if k0 > 0 else xs[lid][0]
            for k in range(k0, n):
                v_common = max(0.0, v_common - sc.post_collision_decel_mps2 / sc.fps)
                x_lead = x_lead + d * v_common / sc.fps
                xs[lid][k] = x_lead
                vs[lid][k] = v_common
                xs[fid][k] = x_lead - d * (vf[lid].length_m / 2.0 + vf[fid].length_m / 2.0)
                vs[fid][k] = v_common

    static_bg = None
    depth = None
    if s.speed_mps == 0 and s.shake_deg == 0:
        static_bg = render_road(cam, sc.road, rng)
        if sc.degradation.fog_visibility_m > 0:
            depth = ground_depth_map(cam)

    video_path = os.path.join(out_dir, f"{sc.name}.mp4")
    writer = None
    if write_video:
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), sc.fps, (s.image_w, s.image_h))

    frames_gt = []
    cam_x = s.x_m
    cam_x_track = None
    cam_v_track = None
    if s.speed_mps > 0:
        cam_spec = VehicleSpec("camera", x0_m=s.x_m, v0_mps=s.speed_mps, lane_y_m=s.y_m,
                               segments=[MotionSegment(a, b) for a, b in (s.accel_segments or [(0.0, 0.0)])])
        cam_x_track, cam_v_track, _ = simulate_vehicle(cam_spec, t)
    for k in range(n):
        if s.speed_mps > 0 or s.shake_deg > 0:
            jitter_p = rng.normal(0, s.shake_deg) if s.shake_deg > 0 else 0.0
            jitter_r = rng.normal(0, s.shake_deg) if s.shake_deg > 0 else 0.0
            cam_x = float(cam_x_track[k]) if cam_x_track is not None else s.x_m
            cam.set_pose(cam_x, s.pitch_deg + jitter_p, s.roll_deg + jitter_r, s.yaw_deg)
            frame = render_road(cam, sc.road, rng)
            if sc.degradation.fog_visibility_m > 0:
                depth = ground_depth_map(cam)
        else:
            frame = static_bg.copy()
        # painter's order: farthest vehicle first
        order = sorted(sc.vehicles, key=lambda v: -np.linalg.norm(np.array([xs[v.vid][k], lateral_offset(v, t[k]), 0]) - cam.c))
        vgt = {}
        for v in order:
            y = float(lateral_offset(v, t[k]))
            bbox = draw_vehicle(frame, cam, xs[v.vid][k], y, v)
            rear = cam.project(np.array([[xs[v.vid][k] - v.direction * v.length_m / 2.0, y, 0.0]]))[0][0]
            front = cam.project(np.array([[xs[v.vid][k] + v.direction * v.length_m / 2.0, y, 0.0]]))[0][0]
            vgt[v.vid] = {"x_m": float(xs[v.vid][k]), "y_m": y, "speed_mps": float(vs[v.vid][k]),
                          "rear_contact_uv": [float(rear[0]), float(rear[1])],
                          "front_contact_uv": [float(front[0]), float(front[1])],
                          "bbox": bbox}
        if s.hood_frac > 0:
            hy = int(s.image_h * (1 - s.hood_frac))
            frame[hy:] = (18, 18, 20)
        frame = apply_degradation(frame, sc.degradation, rng, depth)
        if writer is not None:
            writer.write(frame)
        frames_gt.append({"frame": k, "t_s": float(t[k]), "camera_x_m": float(cam_x),
                          "camera_speed_mps": float(cam_v_track[k]) if cam_v_track is not None else 0.0,
                          "vehicles": vgt})
    if writer is not None:
        writer.release()

    cam_static = Camera(sc.camera)
    gt = {
        "simulator": "evaluation/sim/world.py v2",
        "scenario": asdict(sc),
        "fps": sc.fps, "n_frames": n,
        "collision": ({"t_s": col[0], "follower": col[1], "lead": col[2]} if col else None),
        "free_motion": {k: {"x_m": val[0].tolist(), "speed_mps": val[1].tolist()} for k, val in free.items()},
        "true_world_to_image_H_at_t0": cam_static.ground_homography_world_to_image().tolist(),
        "frames": frames_gt,
    }
    gt_path = os.path.join(out_dir, f"{sc.name}_ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(gt, f)
    return video_path, gt_path, gt
