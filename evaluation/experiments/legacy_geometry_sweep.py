"""
BASELINE (before any algorithm change): how robust is the ORIGINAL geometry
agent (backend/agent/loop.py as received) once the scene stops matching its
assumptions exactly?

Varies, on simulator v2 (evaluation/sim/world.py):
  * camera roll, including non-integer angles (the original stress test used
    exactly 8.0 deg, which the integer-degree search grid can hit exactly);
  * camera yaw (camera not looking exactly along the road);
  * camera height/pitch;
  * TRUE road-marking dimensions vs the pipeline's assumed 3.7 m / 3 m / 9 m;
  * dash phase (random, recorded).

Metric that isolates geometry from tracking: map the vehicle's TRUE image
ground-contact points (from ground truth) through the recovered homography
and compare the recovered along-road displacement over the clip with the true
displacement -> "displacement scale error". Speed error of a constant-speed
vehicle is the same number.
"""
import argparse
import itertools
import os
import time

import cv2
import numpy as np

from common import ROOT, CACHE_DIR, read_frames, save_json  # noqa: F401
from evaluation.sim.world import Scenario, CameraSpec, RoadSpec, VehicleSpec, Degradation, render_scenario
from backend.agent.loop import run_geometry_agent
from cv.homography.ground_plane import image_to_world

ASSUMED = dict(lane_half_width_m=1.85, dash_len_m=3.0, dash_gap_m=9.0)


def rotate_points(uv, angle_deg, w, h):
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    uv = np.asarray(uv, float)
    return (M[:, :2] @ uv.T).T + M[:, 2]


def displacement_scale_error(H, gt, roll_corr_deg, w, h):
    uv = np.array([f["vehicles"]["v"]["rear_contact_uv"] for f in gt["frames"]])
    x_true = np.array([f["vehicles"]["v"]["x_m"] for f in gt["frames"]])
    ok = np.isfinite(uv).all(axis=1) & (uv[:, 1] < h - 2) & (uv[:, 1] > 0) & (uv[:, 0] > 0) & (uv[:, 0] < w)
    uv, x_true = uv[ok], x_true[ok]
    if roll_corr_deg:
        uv = rotate_points(uv, roll_corr_deg, w, h)
    world = image_to_world(H, uv)
    d_est = np.hypot(world[-1, 0] - world[0, 0], world[-1, 1] - world[0, 1])
    d_true = x_true[-1] - x_true[0]
    # local (per-step) scale: catches depth-dependent distortion a global ratio hides
    step_est = np.hypot(np.diff(world[:, 0]), np.diff(world[:, 1]))
    step_true = np.diff(x_true)
    local = step_est / np.maximum(step_true, 1e-6)
    return float(d_est / d_true - 1.0), float(np.percentile(np.abs(local - 1), 90))


def make_scenario(name, roll, yaw, height, pitch, edge_hw, dash_len, dash_gap, phase, seed, speed=20.0):
    return Scenario(
        name=name, seed=seed, fps=30.0, duration_s=4.0,
        camera=CameraSpec(height_m=height, pitch_deg=pitch, roll_deg=roll, yaw_deg=yaw),
        road=RoadSpec(topology="centerline", edge_half_width_m=edge_hw, dash_len_m=dash_len,
                      dash_gap_m=dash_gap, dash_phase_m=phase),
        vehicles=[VehicleSpec("v", x0_m=14.0, v0_mps=speed, lane_y_m=-edge_hw / 2.0)],
        degradation=Degradation(noise_std=4.0))


def configs(quick=False):
    rng = np.random.default_rng(20261016)
    rolls = [0.0, 1.5, 3.0, 4.5, 6.0, 8.0, 10.5]
    out = []
    # A: roll sweep, nominal road, default camera
    for r in rolls:
        out.append(("roll", dict(roll=r, yaw=0.0, height=6.0, pitch=22.0, edge_hw=1.85, dash_len=3.0, dash_gap=9.0)))
    # B: yaw
    for y in [2.0, 4.0, 8.0]:
        out.append(("yaw", dict(roll=0.0, yaw=y, height=6.0, pitch=22.0, edge_hw=1.85, dash_len=3.0, dash_gap=9.0)))
    # C: camera height/pitch
    for hgt, p in [(4.0, 12.0), (8.0, 16.0), (10.0, 28.0)]:
        out.append(("pose", dict(roll=0.0, yaw=0.0, height=hgt, pitch=p, edge_hw=1.85, dash_len=3.0, dash_gap=9.0)))
    # D: true marking dimensions differ from the assumed standard
    for ehw, dl, dg in [(1.75, 3.0, 9.0), (1.95, 3.0, 9.0), (1.85, 3.3, 9.9), (1.85, 2.7, 8.1), (1.85, 4.0, 8.0)]:
        out.append(("prior_mismatch", dict(roll=0.0, yaw=0.0, height=6.0, pitch=22.0, edge_hw=ehw, dash_len=dl, dash_gap=dg)))
    for i, (fam, c) in enumerate(out):
        c["phase"] = float(rng.uniform(0, c["dash_len"] + c["dash_gap"]))
        c["seed"] = 1000 + i
    return out[:4] if quick else out


def run(quick=False):
    rows = []
    for i, (fam, c) in enumerate(configs(quick)):
        name = f"legacy_{fam}_{i:02d}"
        sc = make_scenario(name, **c)
        vp, gp, gt = render_scenario(sc, os.path.join(CACHE_DIR, "legacy_geometry"))
        fps, frames = read_frames(vp)
        h, w = frames[0].shape[:2]
        t0 = time.time()
        ag = run_geometry_agent(frames, w, h, ASSUMED["lane_half_width_m"], ASSUMED["dash_len_m"], ASSUMED["dash_gap_m"])
        dt = time.time() - t0
        row = {"name": name, "family": fam, **c, "accepted": bool(ag.success), "roll_correction_deg": ag.roll_correction_deg,
               "skeptic_disputed": bool(ag.disputed), "agent_seconds": round(dt, 2), "stop_reason": ag.stop_reason}
        if ag.success:
            row["disp_scale_error"], row["local_scale_err_p90"] = displacement_scale_error(ag.H, gt, ag.roll_correction_deg, w, h)
        rows.append(row)
        print(f"{name:28s} acc={row['accepted']!s:5s} roll_corr={ag.roll_correction_deg:+.0f} "
              f"disp_err={row.get('disp_scale_error', float('nan')):+.3f} p90loc={row.get('local_scale_err_p90', float('nan')):.3f} "
              f"disputed={row['skeptic_disputed']} ({dt:.1f}s)", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rows = run(args.quick)
    save_json({"description": __doc__, "rows": rows}, "baseline_legacy_geometry_sweep.json")
