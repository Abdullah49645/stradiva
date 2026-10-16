"""
Geometry v2 on EXACTLY the scenes of the baseline sweep
(legacy_geometry_sweep.py: same configurations, same seeds, same cached
renders), so the before/after comparison is like for like.

Same metric: the vehicle's TRUE image ground-contact points (from ground
truth) are mapped through the recovered homography; the recovered
along-road displacement over the clip is compared with the true one
("displacement scale error"; for a constant-speed vehicle it is the speed
error), plus the 90th percentile of the per-frame local scale error.

Same assumed marking standard as the baseline (edge-to-centre 1.85 m,
dashes 3 m + 9 m). In the "prior_mismatch" family the TRUE markings differ
from that assumption: there, a scale error equal to the mismatch is the
expected, correct behaviour (the cycle length is not identifiable from one
static view), and the question is only whether the Skeptic's
marking-standard check notices when the dash PROPORTIONS differ.

  python3 geometry_sweep_v2.py      -> evaluation/results/geometry_sweep_v2.json
"""
import os
import time

import numpy as np

from common import CACHE_DIR, read_frames, save_json
from legacy_geometry_sweep import ASSUMED, configs, make_scenario

from backend.geometry.road_geometry import GeometryConfig, RoadPrior, estimate_road_geometry, image_to_road
from backend.verification.skeptic import check_held_out_markings, check_marking_standard, check_temporal_stability
from evaluation.sim.world import render_scenario


def disp_error(H, gt, w, h):
    uv = np.array([f["vehicles"]["v"]["rear_contact_uv"] for f in gt["frames"]], float)
    x_true = np.array([f["vehicles"]["v"]["x_m"] for f in gt["frames"]])
    ok = np.isfinite(uv).all(axis=1) & (uv[:, 1] < h - 2) & (uv[:, 1] > 0) & (uv[:, 0] > 0) & (uv[:, 0] < w)
    uv, x_true = uv[ok], x_true[ok]
    world = image_to_road(H, uv)
    d_est = np.hypot(world[-1, 0] - world[0, 0], world[-1, 1] - world[0, 1])
    local = np.hypot(np.diff(world[:, 0]), np.diff(world[:, 1])) / np.maximum(np.diff(x_true), 1e-6)
    return float(d_est / (x_true[-1] - x_true[0]) - 1.0), float(np.percentile(np.abs(local - 1), 90))


def main():
    prior = RoadPrior(edge_to_centre_m=ASSUMED["lane_half_width_m"], dash_len_m=ASSUMED["dash_len_m"],
                      dash_gap_m=ASSUMED["dash_gap_m"])
    rows = []
    for i, (fam, c) in enumerate(configs()):
        name = f"legacy_{fam}_{i:02d}"
        sc = make_scenario(name, **c)
        vp, gp, gt = render_scenario(sc, os.path.join(CACHE_DIR, "legacy_geometry"))
        fps, frames = read_frames(vp)
        h, w = frames[0].shape[:2]
        t0 = time.time()
        geo = estimate_road_geometry(frames, prior, GeometryConfig(seed=0))
        row = {"name": name, "family": fam, **c, "accepted": bool(geo.ok), "reasons": geo.reasons,
               "roll_est_deg": geo.roll_deg, "seconds": round(time.time() - t0, 2)}
        if geo.ok:
            row["disp_scale_error"], row["local_scale_err_p90"] = disp_error(geo.H, gt, w, h)
            probe = np.array([[w / 2, h * 0.6], [w / 2, h * 0.95]])
            checks = [check_held_out_markings(geo), check_temporal_stability(frames, geo, prior, probe),
                      check_marking_standard(geo)]
            row["skeptic"] = {ch["check"]: ch["status"] for ch in checks}
            row["skeptic_disputed"] = [ch["check"] for ch in checks if ch["status"] == "DISPUTE"]
            true_cycle = c["dash_len"] + c["dash_gap"]
            row["expected_scale_error_from_prior"] = float((ASSUMED["dash_len_m"] + ASSUMED["dash_gap_m"]) / true_cycle - 1)
        rows.append(row)
        print(f"{name:28s} ok={row['accepted']!s:5s} roll={row['roll_est_deg'] if row['roll_est_deg'] is None else round(row['roll_est_deg'], 2)} "
              f"(true {c['roll']}) disp_err={row.get('disp_scale_error', float('nan')):+.4f} "
              f"disputed={row.get('skeptic_disputed')}", flush=True)
    save_json({"description": __doc__, "assumed": ASSUMED, "rows": rows}, "geometry_sweep_v2.json")


if __name__ == "__main__":
    main()
