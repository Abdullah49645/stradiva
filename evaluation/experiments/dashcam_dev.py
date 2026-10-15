"""
Development check of backend/dashcam/ego_speed.py on SYNTHETIC moving-camera
clips (never on the real comma.ai footage, which is reserved for the
pre-registered evaluation in evaluation/real/). Varies speed, acceleration,
roll/pitch, marking cycle, wear, lane-change-free traffic in adjacent lanes.
Used to choose the estimator's gating thresholds.
"""
import os
import sys
import numpy as np
from common import CACHE_DIR, save_json
from evaluation.sim.world import Scenario, CameraSpec, RoadSpec, VehicleSpec, MotionSegment, Degradation, render_scenario
from backend.dashcam.ego_speed import estimate_ego_speed, MarkingPrior


def scenes():
    rng = np.random.default_rng(77)
    out = []
    for i in range(10):
        v0 = float(rng.uniform(6, 30))
        acc = float(rng.choice([0.0, 0.0, -1.5, 1.0]))
        cyc = float(rng.choice([12.0, 14.63]))
        cam = CameraSpec(image_w=640, image_h=480, focal_px=float(rng.uniform(420, 560)), height_m=float(rng.uniform(1.15, 1.4)),
                         pitch_deg=float(rng.uniform(1.5, 5.0)), roll_deg=float(rng.uniform(-2, 2)), yaw_deg=float(rng.uniform(-1.5, 1.5)),
                         y_m=float(rng.uniform(-0.4, 0.4)), speed_mps=v0, accel_segments=[(0.0, acc)], hood_frac=float(rng.uniform(0.0, 0.15)),
                         shake_deg=float(rng.uniform(0.0, 0.15)))
        road = RoadSpec(topology="multilane", lane_width_m=3.66, n_lanes=3, dash_len_m=cyc * 0.25, dash_gap_m=cyc * 0.75,
                        dash_phase_m=float(rng.uniform(0, cyc)), wear_prob=float(rng.uniform(0, 0.15)), x_max_m=400)
        others = [VehicleSpec(f"o{j}", x0_m=float(rng.uniform(15, 60)), v0_mps=v0 + float(rng.uniform(-3, 3)),
                              lane_y_m=float(rng.choice([-3.66, 3.66])), color_bgr=(60, 60, 160)) for j in range(int(rng.integers(0, 3)))]
        sc = Scenario(name=f"dash_{i:02d}", seed=300 + i, fps=20.0, duration_s=6.0, camera=cam, road=road, vehicles=others,
                      degradation=Degradation(noise_std=float(rng.uniform(2, 5)), blur_px=float(rng.uniform(0, 0.8))))
        out.append((sc, cyc))
    return out


def main():
    rows = []
    for sc, cyc in scenes():
        vp, gp, gt = render_scenario(sc, os.path.join(CACHE_DIR, "dashcam_dev"))
        res = estimate_ego_speed(vp, MarkingPrior("true cycle (dev)", cyc))
        tt = np.array([f["t_s"] for f in gt["frames"]])
        vv = np.array([f["camera_speed_mps"] for f in gt["frames"]])
        for w in res["windows"]:
            m = (tt >= w["t_start_s"]) & (tt < w["t_end_s"])
            v_true = float(vv[m].mean())
            row = {"scene": sc.name, "t0": w["t_start_s"], "true": v_true, "status": w["status"], "est": w["speed_mps"],
                   "ci": w["ci90_mps"], "reasons": w["reasons"]}
            if w["status"] == "OK":
                row["err_pct"] = 100 * (w["speed_mps"] - v_true) / v_true
                row["covered"] = bool(w["ci90_mps"][0] <= v_true <= w["ci90_mps"][1])
            rows.append(row)
            print(sc.name, f"{w['t_start_s']:.0f}s true={v_true:5.2f} {w['status']:7s} est={w['speed_mps'] if w['speed_mps'] is None else round(w['speed_mps'],2)} "
                  f"err={row.get('err_pct', float('nan')):+.2f}% cov={row.get('covered')} {'; '.join(w['reasons'])[:80]}", flush=True)
    ok = [r for r in rows if r["status"] == "OK"]
    if ok:
        e = np.array([abs(r["err_pct"]) for r in ok])
        print(f"answered {len(ok)}/{len(rows)}; median |err| {np.median(e):.2f}%  p90 {np.percentile(e,90):.2f}%  coverage {np.mean([r['covered'] for r in ok]):.0%}")
    save_json({"rows": rows}, "dashcam_dev.json")


if __name__ == "__main__":
    main()
