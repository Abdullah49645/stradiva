"""
Builds the case files bundled with the web UI (frontend/public/cases/).

Every case is produced by the unchanged production pipeline
(backend.pipeline.analyze.analyze_video) on a video file -- the pipeline never
sees ground truth. Ground truth is attached AFTERWARDS, as a separate
"validation" table the UI labels as such, only for synthetic footage where it
exists.

Cases:
  rear_end_roadside   synthetic, roadside pole camera, lead brakes hard, follower brakes late
  no_scale            synthetic, the same kind of road without a dashed line: no metric scale -> refusal
  bumped_camera       synthetic, the camera is knocked 1.5 deg mid-clip: the Skeptic disputes, verdict withheld
  dashcam_real        REAL comma2k19 segment (MIT licence): moving camera -> own-vehicle speed with the
                      lane-width / focal-length scale cross-check

Run from the repo root:  python3 evaluation/demo/build_cases.py
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation", "experiments"))

from evaluation.sim.world import (Scenario, CameraSpec, RoadSpec, VehicleSpec, MotionSegment, Degradation,  # noqa: E402
                                  render_scenario)
from backend.pipeline.analyze import analyze_video  # noqa: E402

OUT = os.path.join(ROOT, "frontend", "public", "cases")
WORK = os.path.join(ROOT, "evaluation", "_sim_cache", "demo")
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


def to_webm(src, dst, width=960, fps=None, bitrate="900k"):
    """Evidence clip for the browser (VP8 WebM). Frames are decoded with OpenCV (so the clip is exactly
    what the pipeline read, at the frame rate it used) and piped to the ffmpeg CLI for a size-bounded
    encode; without ffmpeg, OpenCV's own VideoWriter is used (larger files)."""
    import shutil
    import subprocess
    cap = cv2.VideoCapture(src)
    f0 = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, fr = cap.read()
    h0, w0 = fr.shape[:2]
    scale = width / w0
    size = (width, int(round(h0 * scale / 2) * 2))
    if shutil.which("ffmpeg"):
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                              "-s", f"{size[0]}x{size[1]}", "-r", f"{f0}", "-i", "-", "-c:v", "libvpx",
                              "-b:v", bitrate, "-an", dst], stdin=subprocess.PIPE)
        while ok:
            p.stdin.write(cv2.resize(fr, size, interpolation=cv2.INTER_AREA).tobytes())
            ok, fr = cap.read()
        p.stdin.close()
        p.wait()
        return
    wr = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"VP80"), f0, size)
    while ok:
        wr.write(cv2.resize(fr, size, interpolation=cv2.INTER_AREA))
        ok, fr = cap.read()
    wr.release()


def rear_end_scene(name, road_kw=None):
    """The illustrative case. Chosen BEFORE running CARAI on it, from ground-truth properties only
    (search grid in docs/demo_plan.md): the first candidate in which the collision happens 28-48 m
    from the camera (where one image row spans < 0.3 m of road), the lead brakes >= 1 s before
    contact, and the true verdict is ASSUMPTION_DEPENDENT. CARAI's output on it is reported as is;
    the aggregate results are in the "How it was tested" tab, not here."""
    rk = dict(topology="centerline", edge_half_width_m=1.82, dash_len_m=3.05, dash_gap_m=9.1, dash_phase_m=4.2, wear_prob=0.05)
    rk.update(road_kw or {})
    return Scenario(
        name=name, seed=4242, fps=30.0, duration_s=6.0,
        camera=CameraSpec(height_m=8.0, pitch_deg=16.0, roll_deg=2.3, yaw_deg=7.5, x_m=-12.0, y_m=-6.5, focal_px=960.0),
        road=RoadSpec(**rk),
        vehicles=[VehicleSpec("lead", x0_m=22.0, v0_mps=10.0, lane_y_m=-0.95, length_m=4.6, color_bgr=(160, 120, 40),
                              segments=[MotionSegment(0, 0.0), MotionSegment(0.6, -7.5)]),
                  VehicleSpec("follow", x0_m=3.45, v0_mps=12.0, lane_y_m=-0.95, length_m=4.5, color_bgr=(40, 40, 185),
                              segments=[MotionSegment(0, 0.0), MotionSegment(1.8, -4.0)])],
        degradation=Degradation(noise_std=3.5, blur_px=0.5),
        notes="lead brakes at 0.6 s (7.5 m/s^2); follower reacts 1.2 s later and brakes at only 4 m/s^2")


def gt_validation(sc, gt, case):
    sys.path.insert(0, os.path.join(ROOT, "evaluation", "experiments"))
    from collision_suite import gt_counterfactual
    meta = {"hazard_onset_s": next(s.t_start_s for s in sc.vehicles[0].segments if s.accel_mps2 < 0)}
    col = gt.get("collision")
    g = gt_counterfactual(sc, meta, col["t_s"] if col else None)
    rows = []
    rows.append({"label": "Verdict over the assumption box", "truth": g["verdict"],
                 "carai": (case.get("verdict") or {}).get("value", case["status"]),
                 "ok": g["verdict"] == (case.get("verdict") or {}).get("value")})
    cr = (case.get("counterfactual") or {}).get("critical_reaction_s", {}).get("7")
    if cr:
        t7 = g["critical_reaction_s"]["7"]
        rows.append({"label": "Latest avoiding reaction at 7 m/s²", "truth": f"{t7:.2f} s",
                     "carai": f"{cr['median_s']:.2f} s ({cr['ci90_s'][0]:.2f}–{cr['ci90_s'][1]:.2f})",
                     "ok": cr["ci90_s"][0] - 0.05 <= t7 <= cr["ci90_s"][1] + 0.05})
    hz = case.get("hazard_onset") or {}
    if hz.get("source") == "lead_brake_onset":
        rows.append({"label": "Lead brake onset", "truth": f"{meta['hazard_onset_s']:.2f} s", "carai": f"{hz['t_s']:.2f} s",
                     "ok": abs(hz["t_s"] - meta["hazard_onset_s"]) < 0.15})
    if col and (case.get("collision") or {}).get("t_contact_s") is not None:
        rows.append({"label": "Contact time", "truth": f"{col['t_s']:.2f} s", "carai": f"{case['collision']['t_contact_s']:.2f} s",
                     "ok": abs(case["collision"]["t_contact_s"] - col["t_s"]) < 0.1})
    fs = case.get("follower_speed_at_hazard_onset")
    if fs:
        k = int(round(meta["hazard_onset_s"] * gt["fps"]))
        vt = gt["frames"][k]["vehicles"]["follow"]["speed_mps"]
        rows.append({"label": "Follower speed at hazard onset", "truth": f"{vt * 3.6:.1f} km/h",
                     "carai": f"{fs['ci90_mps'][0] * 3.6:.1f}–{fs['ci90_mps'][1] * 3.6:.1f} km/h",
                     "ok": fs["ci90_mps"][0] <= vt <= fs["ci90_mps"][1]})
    return {"note": "Ground truth exists only because this footage is synthetic. It was not available to the pipeline; "
                    "it is compared here after the fact.", "rows": rows}


def build_synthetic(cid, sc, title, summary, post=None):
    vp, gp, gt = render_scenario(sc, WORK)
    if post:
        vp = post(vp)
    case = analyze_video(vp, seed=0)
    case["id"] = cid
    case["provenance"] = {"kind": "synthetic", "title": title, "summary": summary,
                          "source": "CARAI simulator (evaluation/sim/world.py)"}
    case["validation"] = gt_validation(sc, gt, case)
    to_webm(vp, os.path.join(OUT, f"{cid}.webm"))
    case["video"] = f"/cases/{cid}.webm"
    with open(os.path.join(OUT, f"{cid}.json"), "w") as f:
        json.dump(case, f)
    print(cid, case["status"], (case.get("verdict") or {}).get("value"), [r["ok"] for r in case["validation"]["rows"]])
    return {"id": cid, "title": title, "summary": summary}


def bump(vp):
    from skeptic_faults import bump_video
    out = vp.replace(".mp4", "_bumped.mp4")
    bump_video(vp, out, 30.0)
    return out


def build_dashcam_real(seg_dir):
    from backend.dashcam.ego_speed import estimate_ego_speed, MarkingPrior, EgoSpeedConfig
    video = os.path.join(seg_dir, "video.hevc")
    res = estimate_ego_speed(video, MarkingPrior.preset("california_freeway"), fps_override=20.0,
                             cfg=EgoSpeedConfig(focal_px=910.0, lane_width_prior_m=3.6))
    ft = np.load(os.path.join(seg_dir, "global_pose", "frame_times"))
    ct = np.load(os.path.join(seg_dir, "processed_log", "CAN", "speed", "t"))
    cvv = np.load(os.path.join(seg_dir, "processed_log", "CAN", "speed", "value")).ravel()
    vf = np.interp(ft, ct, cvv)
    for w in res["windows"]:
        a, b = int(round(w["t_start_s"] * 20)), int(round(w["t_end_s"] * 20))
        w["truth_mps"] = float(vf[a:b].mean())
    implied = [w["diagnostics"].get("scale_cross_check", {}).get("implied_cycle_m") for w in res["windows"]]
    implied = [x for x in implied if x]
    case = {"id": "dashcam_real", "status": "MOVING_CAMERA", "pipeline_version": "carai-2.0",
            "verdict": {"value": "MOVING_CAMERA",
                        "reason": ("The camera is moving, so CARAI measures the camera vehicle's own speed from the lane "
                                   "markings sliding past. Assuming freeway markings (48 ft cycle), lane width and the "
                                   "camera's documented focal length disagree with that assumption in "
                                   f"{sum('inconsistent with lane geometry' in ' '.join(w['reasons']) for w in res['windows'])} "
                                   f"of {len(res['windows'])} windows, so CARAI refuses rather than report speeds that, if the "
                                   f"lane geometry is right, would be {14.63 / np.median(implied):.1f}x too high. "
                                   f"The lane geometry implies a marking cycle of "
                                   f"{np.median(implied):.1f} m, which is a city-street pattern, not a freeway one.")},
            "input": {"file": "comma2k19 Example_1 segment 40 (video.hevc)", "fps": 20.0, "width": 1164, "height": 874,
                      "fps_note": "container reports 25 fps; sensor timestamps show 20 fps, which is used"},
            "dashcam": {"fps": 20.0, "fps_source": res["fps_source"], "prior": res["prior"],
                        "windows": [{k: w[k] for k in ("t_start_s", "t_end_s", "status", "speed_mps", "ci90_mps", "reasons", "truth_mps")}
                                    for w in res["windows"]]},
            "provenance": {"kind": "real", "title": "Real dashcam, California",
                           "summary": "One minute of real driving with the car's own CAN-bus speed for comparison.",
                           "source": "comma2k19 (comma.ai, MIT licence)", "license": "MIT"}}
    # a second pass with the marking cycle the lane geometry implies, shown as a comparison
    res2 = estimate_ego_speed(video, MarkingPrior("cycle implied by lane geometry", float(np.median(implied))), fps_override=20.0,
                              cfg=EgoSpeedConfig(focal_px=910.0, lane_width_prior_m=3.6))
    ok2 = [(w, float(vf[int(round(w['t_start_s'] * 20)):int(round(w['t_end_s'] * 20))].mean())) for w in res2["windows"] if w["status"] == "OK"]
    ape = [abs(w["speed_mps"] - t) / t for w, t in ok2]
    cov = [w["ci90_mps"][0] <= t <= w["ci90_mps"][1] for w, t in ok2]
    case["validation"] = {"note": "The car's CAN-bus speed was not available to CARAI. Post-hoc comparison: this segment "
                                  "was the pre-registered holdout, where the freeway assumption failed; see 'Real dashcam footage'.",
                          "rows": [{"label": "Windows answered with the freeway assumption", "truth": "–",
                                    "carai": f"{sum(w['status'] == 'OK' for w in res['windows'])} of {len(res['windows'])}", "ok": None},
                                   {"label": "Marking cycle implied by lane geometry", "truth": "unknown (city street)",
                                    "carai": f"{np.median(implied):.2f} m", "ok": None},
                                   {"label": "Speed error using the implied cycle (median)", "truth": "CAN bus",
                                    "carai": f"{100 * np.median(ape):.1f} % over {len(ok2)} windows", "ok": None},
                                   {"label": "90 % intervals containing the CAN speed", "truth": "–",
                                    "carai": f"{sum(cov)} of {len(cov)}", "ok": None}]}
    to_webm(video, os.path.join(OUT, "dashcam_real.webm"), width=720, fps=20.0, bitrate="600k")
    case["video"] = "/cases/dashcam_real.webm"
    with open(os.path.join(OUT, "dashcam_real.json"), "w") as f:
        json.dump(case, f)
    print("dashcam_real", sum(w["status"] == "OK" for w in res["windows"]), "answered")
    return {"id": "dashcam_real", "title": "Real dashcam, wrong assumption caught",
            "summary": "Real footage; CARAI refuses a marking assumption the geometry contradicts."}


if __name__ == "__main__":
    index = []
    index.append(build_synthetic("rear_end_roadside", rear_end_scene("demo_rear_end"), "Rear-end, roadside camera",
                                 "Lead brakes hard; the follower reacts late. Synthetic, with known ground truth."))
    index.append(build_synthetic("no_scale", rear_end_scene("demo_no_scale", {"dash_len_m": 0.0001, "dash_gap_m": 12.0}),
                                 "No dashed line", "The road has no marking of known length, so CARAI refuses to measure."))
    index.append(build_synthetic("bumped_camera", rear_end_scene("demo_bumped"), "Camera knocked mid-clip",
                                 "The pole camera moves 1.5° halfway through; the cross-examination catches it.", post=bump))
    seg = os.environ.get("CARAI_COMMA2K19_SEG", os.path.join(ROOT, "evaluation", "real", "data", "comma2k19", "seg40"))
    if os.path.exists(os.path.join(seg, "video.hevc")):
        index.append(build_dashcam_real(seg))
    else:
        print("comma2k19 segment not found; skipping the real dashcam case")
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
