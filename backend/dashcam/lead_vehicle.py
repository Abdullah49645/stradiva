"""
Dashcam: range, relative speed and time-to-collision of the vehicle ahead.

Pre-registered test: evaluation/real/PREREGISTRATION_LEAD.md (comma2k19, radar ground truth).

Method
------
1. Vehicles: YOLOX-S (OpenCV Model Zoo, Apache-2.0) on OpenCV 5 DNN
   (cv/detection/dnn_detector.py), every `det_stride`-th frame.
2. Road geometry per window of `window_s`: road vanishing point and lane-line
   directions about it (the Hough + RANSAC stage of backend/dashcam/ego_speed.py).
   The ego lane is bounded by the nearest lines left and right of vertical.
3. Camera pitch theta from the vanishing point. Camera height h from the ego-lane
   geometry, h = W cos(theta) / (tan a_R - tan a_L), with the lane-width prior
   W = 3.6 m +- 10 %. This is the ONLY metric assumption. h is a property of the
   mounting, so the clip median is used.
4. Lead: the detection whose bottom-centre lies inside the ego lane at its own
   row and is nearest the camera. It is followed by box overlap; a different
   vehicle starts a new segment.
5. Range from the flat road under the lead's tyres (window median):
       Z = h / tan(theta + atan((v_bottom - c_y) / f))
   Relative speed from the box's expansion rate, dZ/dt = -Z d(ln w)/dt
   (Theil-Sen slope over the window; insensitive to pitch jitter, which moves
   the box bottom by pixels but not its width). A second, independent ruler is
   the box width with a vehicle-width prior by class (car 1.8 m, truck/bus
   2.5 m, +-8 %). The two ranges are fused by inverse variance -- the
   ground-plane error grows with range (vanishing-point jitter measured in the
   window), the width error does not -- and a window where they disagree
   beyond their combined 2-sigma is refused.
   TTC = Z / closing speed, reported only while the lower 90 % bound of the
   closing speed exceeds 0.5 m/s.
6. Intervals: bootstrap over the window's frames combined with Monte Carlo
   over the lane-width prior. All ranges and speeds scale linearly with it.

What it cannot do: the road must be locally flat between the camera and the
lead; a box bottom that does not touch the road (e.g. a truck's raised rear)
reads farther than true; the focal length must be known.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.dashcam.ego_speed import EgoSpeedConfig, _line_candidates, _segments, _vanishing_point, detect_static_bottom
from cv.detection.dnn_detector import get_detector


@dataclass
class LeadConfig:
    focal_px: float = 910.0
    lane_width_prior_m: float = 3.6
    lane_width_rel_sigma: float = 0.10
    window_s: float = 2.0
    det_stride: int = 1
    bottom_sigma_px: float = 2.0
    min_lead_frac: float = 0.5          # a window needs the lead in this fraction of its detector frames
    closing_min_mps: float = 0.5
    max_range_m: float = 120.0
    n_mc: int = 400
    max_gap_frames: int = 10            # a lead missed for up to this many frames keeps its identity
    vehicle_width_prior_m: dict = None  # second ruler, by detector class
    vehicle_width_rel_sigma: float = 0.08
    boot_block_s: float = 0.5

    def __post_init__(self):
        if self.vehicle_width_prior_m is None:
            self.vehicle_width_prior_m = {"car": 1.8, "truck": 2.5, "bus": 2.55}
    lead_min_score: float = 0.5
    bottom_exclude_frac: float = 0.85   # boxes ending in the bottom 15 % of the frame (bonnet, dashboard)


def _window_geometry(grays, cfg: LeadConfig, v_limit, rng):
    ec = EgoSpeedConfig(focal_px=cfg.focal_px, lane_width_prior_m=cfg.lane_width_prior_m)
    vps, angs = [], []
    for g in grays[:: max(1, len(grays) // 10)]:
        segs = _segments(g, ec, v_limit)
        vp, inl = _vanishing_point(segs, ec, rng)
        if vp is None:
            continue
        vps.append(vp)
        angs.append(_line_candidates(segs, inl, vp, max_lines=6))
    if len(vps) < 3:
        return None, "no stable road vanishing point"
    vp_arr = np.array(vps)
    vp = np.median(vp_arr, axis=0)
    vp_row_sd = float(1.4826 * np.median(np.abs(vp_arr[:, 1] - vp[1])))
    left = [a for aa in angs for a in aa if -60 < a < -3]
    right = [a for aa in angs for a in aa if 3 < a < 60]
    if not left or not right:
        return None, "ego-lane boundaries not both visible"
    # ego lane: the lines nearest vertical on each side, as the median of each frame's nearest
    aL = float(np.median([max([a for a in aa if -60 < a < -3], default=np.nan) for aa in angs]))
    aR = float(np.median([min([a for a in aa if 3 < a < 60], default=np.nan) for aa in angs]))
    if not (np.isfinite(aL) and np.isfinite(aR)):
        return None, "ego-lane boundaries not both visible"
    h_img = grays[0].shape[0]
    theta = float(np.arctan((h_img / 2.0 - vp[1]) / cfg.focal_px))
    dtan = np.tan(np.radians(aR)) - np.tan(np.radians(aL))
    h_cam = float(cfg.lane_width_prior_m * np.cos(theta) / dtan)
    return {"vp": vp.tolist(), "aL": aL, "aR": aR, "theta_rad": theta, "h_cam_m": h_cam, "vp_row_sd_px": vp_row_sd}, None


def _in_lane(geo, u, v):
    if v <= geo["vp"][1] + 2:
        return False
    uL = geo["vp"][0] + (v - geo["vp"][1]) * np.tan(np.radians(geo["aL"]))
    uR = geo["vp"][0] + (v - geo["vp"][1]) * np.tan(np.radians(geo["aR"]))
    return uL < u < uR


def _range(v_bottom, theta, h_cam, f, cy):
    ang = theta + np.arctan((v_bottom - cy) / f)
    return np.where(ang > 1e-4, h_cam / np.tan(np.maximum(ang, 1e-4)), np.inf)


def analyze_lead(video_path, fps, cfg: LeadConfig = None, max_frames=None, seed=0):
    cfg = cfg or LeadConfig()
    det = get_detector()
    if det is None:
        return {"status": "NOT_AVAILABLE", "reason": "vehicle detector model not installed (models/fetch_models.sh)"}
    rng = np.random.default_rng(seed)
    cap = cv2.VideoCapture(video_path)
    frames_gray, dets, k = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and k >= max_frames):
            break
        frames_gray.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        dets.append(det.detect(f) if k % cfg.det_stride == 0 else None)
        k += 1
    cap.release()
    n = len(frames_gray)
    if n < 10:
        return {"status": "INVALID_INPUT", "reason": "too few frames"}
    hgt, wid = frames_gray[0].shape
    cy = hgt / 2.0
    v_limit = detect_static_bottom(frames_gray)
    win = int(round(cfg.window_s * fps))
    geos = []
    for s in range(0, n, win):
        g, why = _window_geometry(frames_gray[s:s + win], cfg, v_limit, rng)
        geos.append((s, min(n, s + win), g, why))
    good = [g for _, _, g, _ in geos if g]
    if not good:
        return {"status": "NOT_RECONSTRUCTABLE", "reason": "no window with a usable ego lane", "windows": []}
    h_cam = float(np.median([g["h_cam_m"] for g in good]))
    # the camera is rigidly mounted: where a window's lane lines are not both visible, use the clip's
    # median lane geometry (stated in the window's notes) rather than discarding the window
    clip_geo = {"vp": np.median([g["vp"] for g in good], axis=0).tolist(), "aL": float(np.median([g["aL"] for g in good])),
                "aR": float(np.median([g["aR"] for g in good])), "theta_rad": float(np.median([g["theta_rad"] for g in good])),
                "h_cam_m": h_cam, "vp_row_sd_px": float(np.median([g["vp_row_sd_px"] for g in good]))}
    geos = [(s, e, g if g else clip_geo, None if g else "clip-median lane geometry used (" + why + " in this window)")
            for s, e, g, why in geos]

    # lead per frame: nearest detection inside the ego lane, linked by box overlap across short gaps
    lead = [None] * n
    seg_id, last, last_i = 0, None, -999
    for s, e, g, _ in geos:
        for i in range(s, e):
            if dets[i] is None:
                continue
            cands = []
            for d in dets[i]:
                x, y, bw, bh = d["bbox"]
                u, v = x + bw / 2.0, y + bh
                if v >= min(v_limit, cfg.bottom_exclude_frac * hgt) - 2 or d["score"] < cfg.lead_min_score \
                        or not _in_lane(g, u, v):
                    continue      # bonnet / dashboard reflections and weak detections are not a lead
                cands.append((v, d))
            if not cands:
                continue
            v, d = max(cands, key=lambda c: c[0])
            new_seg = True
            if last is not None and i - last_i <= cfg.max_gap_frames:
                px, py, pw, ph = last["bbox"]
                x, y, bw, bh = d["bbox"]
                ix = max(0, min(px + pw, x + bw) - max(px, x))
                iy = max(0, min(py + ph, y + bh) - max(py, y))
                new_seg = ix * iy / max(1e-9, pw * ph + bw * bh - ix * iy) < 0.2
            if new_seg:
                seg_id += 1
            lead[i] = {"bbox": d["bbox"], "score": d["score"], "seg": seg_id, "cls": d["cls"],
                       "v_bottom": d["bbox"][1] + d["bbox"][3], "width_px": d["bbox"][2], "theta": g["theta_rad"]}
            last, last_i = d, i

    t_all = np.arange(n) / fps
    frames_out = []
    for i, l in enumerate(lead):
        if l:
            Z = float(_range(np.array([l["v_bottom"]]), l["theta"], h_cam, cfg.focal_px, cy)[0])
            frames_out.append({"t_s": float(t_all[i]), "seg": l["seg"], "range_ground_m": Z, "width_px": l["width_px"],
                               "bbox": l["bbox"], "cls": l["cls"]})

    # windows: range = median ground-plane range; relative speed from the box's expansion rate,
    # dZ/dt = -Z d(ln w)/dt (independent of pitch jitter); a second ruler (typical car width) cross-checks Z
    scale = 1.0 + rng.normal(0, cfg.lane_width_rel_sigma, cfg.n_mc)
    windows = []
    for s, e, g, note in geos:
        fr = [f for f in frames_out if s / fps <= f["t_s"] < e / fps]
        det_frames = sum(1 for i in range(s, e) if dets[i] is not None)
        w = {"t_start_s": s / fps, "t_end_s": e / fps, "status": "ABSTAIN", "reasons": [], "notes": [note] if note else []}
        if fr:
            segs_ = [f["seg"] for f in fr]
            main = max(set(segs_), key=segs_.count)
            fr = [f for f in fr if f["seg"] == main]
        if len(fr) < cfg.min_lead_frac * max(1, det_frames):
            w["reasons"].append(f"lead vehicle in the ego lane tracked in {len(fr)} of {det_frames} frames")
            windows.append(w)
            continue
        Zf = np.array([f["range_ground_m"] for f in fr])
        Zf = Zf[np.isfinite(Zf)]
        if len(Zf) < 6 or np.median(Zf) > cfg.max_range_m:
            w["reasons"].append("lead too far or below the horizon")
            windows.append(w)
            continue
        tt = np.array([f["t_s"] for f in fr])
        lw = np.log(np.array([f["width_px"] for f in fr]))
        Zm = float(np.median(Zf))
        boot_s, boot_z = [], []
        # moving-block bootstrap: box jitter is correlated over consecutive frames, so resampling single
        # frames would understate the uncertainty of the slope
        blk = max(2, int(round(cfg.boot_block_s * fps / cfg.det_stride)))
        for _ in range(200):
            starts = rng.integers(0, max(1, len(tt) - blk + 1), int(np.ceil(len(tt) / blk)))
            b = np.concatenate([np.arange(st, min(len(tt), st + blk)) for st in starts])[: len(tt)]
            if np.ptp(tt[b]) < 0.5:
                continue
            boot_s.append(_theil_sen(tt[b], lw[b]))
            boot_z.append(np.median(Zf[rng.integers(0, len(Zf), len(Zf))]))
        if len(boot_s) < 50:
            w["reasons"].append("lead tracked for too short a time in this window")
            windows.append(w)
            continue
        slope = _theil_sen(tt, lw)
        v_rel = -Zm * slope
        bs, bz = np.array(boot_s), np.array(boot_z)
        k = rng.integers(0, len(bs), cfg.n_mc)
        cls = max(set(f["cls"] for f in fr), key=[f["cls"] for f in fr].count)
        w_m = float(np.median([f["width_px"] for f in fr]))
        # two independent rulers. Ground plane: relative error from the row uncertainty (vanishing-point jitter
        # in this window + box-bottom noise) grows with range, plus the lane-width prior. Vehicle width: a
        # class prior (cars 1.8 m, trucks/buses 2.5 m, +-8 %), constant relative error. Fused by inverse variance.
        sig_row = float(np.hypot(g.get("vp_row_sd_px", 2.0), cfg.bottom_sigma_px))
        rel_g = float(np.hypot(Zm * sig_row / (cfg.focal_px * h_cam), cfg.lane_width_rel_sigma))
        w_prior = cfg.vehicle_width_prior_m.get(cls)
        Zw = cfg.focal_px * w_prior / w_m if w_prior else None
        if Zw is not None:
            rel_w = cfg.vehicle_width_rel_sigma
            wg, ww = 1 / rel_g ** 2, 1 / rel_w ** 2
            Zf_ = float(np.exp((wg * np.log(Zm) + ww * np.log(Zw)) / (wg + ww)))
            rel_f = float(np.sqrt(1 / (wg + ww)))
            disagree = abs(np.log(Zm / Zw)) > 2 * np.hypot(rel_g, rel_w)
        else:
            Zf_, rel_f, disagree = Zm, rel_g, False
        bz_rel = bz / np.median(bz)                      # bootstrap spread of the window median (ground plane)
        Zs = Zf_ * np.exp(rng.normal(0, rel_f, cfg.n_mc)) * bz_rel[k] ** (rel_f ** 2 / rel_g ** 2)
        vs = -Zs * bs[k]
        v_rel = -Zf_ * slope
        w.update({"status": "OK", "range_m": Zf_, "range_ci90_m": [float(np.percentile(Zs, 5)), float(np.percentile(Zs, 95))],
                  "rel_speed_mps": float(v_rel), "rel_speed_ci90_mps": [float(np.percentile(vs, 5)), float(np.percentile(vs, 95))],
                  "n_frames": len(fr), "range_ground_plane_m": Zm, "range_ground_plane_rel_sigma": rel_g,
                  "range_from_width_m": Zw, "lead_class": cls})
        if disagree:
            w.update({"status": "ABSTAIN"})
            w["reasons"].append(f"ground-plane range {Zm:.0f} m and vehicle-width range {Zw:.0f} m disagree beyond their "
                                "combined uncertainty; one of the assumptions (flat road, lane width, vehicle width) is wrong")
            windows.append(w)
            continue
        closing = -vs
        if np.percentile(closing, 5) > cfg.closing_min_mps:
            ttc = Zs / closing
            w["ttc_s"] = float(np.median(ttc))
            w["ttc_ci90_s"] = [float(np.percentile(ttc, 5)), float(np.percentile(ttc, 95))]
        windows.append(w)
    return {"status": "OK", "camera_height_m": h_cam, "camera_height_note":
            f"from lane geometry with a {cfg.lane_width_prior_m} m +- {100 * cfg.lane_width_rel_sigma:.0f} % lane-width prior",
            "focal_px": cfg.focal_px, "fps": fps, "detector": "YOLOX-S (OpenCV Model Zoo) on OpenCV DNN",
            "windows": windows, "frames": frames_out}


def _theil_sen(t, y):
    """Median of pairwise slopes (robust to detector box jitter and outliers)."""
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]
    ok = dt > 0.15
    if not ok.any():
        return 0.0
    return float(np.median((y[j][ok] - y[i][ok]) / dt[ok]))
