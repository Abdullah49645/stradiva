"""
Multi-vehicle tracking for a static camera (replaces the "two largest blobs,
sorted by image row" association in cv/tracking/motion_tracker.py, which has
no identity, breaks under occlusion and assumes exactly two vehicles).

Detection: MOG2 background subtraction seeded from the clip's temporal-median
background (no ground truth, no known camera model), shadow pixels dropped,
morphological clean-up, connected components.

Association: SORT-style. One cv2.KalmanFilter per track over the bounding
box (centre, size, centre velocity); Hungarian assignment
(scipy.optimize.linear_sum_assignment) on 1 - IoU with a gate.

Occlusion / merging: if one detection overlaps the predictions of two
tracks, the blob is the two vehicles fused together -- its bottom edge is
not either vehicle's ground contact. Neither track is updated from it
(both coast and are marked MERGED for that frame). This is what keeps the
last metres before a rear-end impact from being silently corrupted.

Ground contact: bottom-centre of the box. Boxes touching the image border
are flagged (contact point off-screen).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackerConfig:
    min_area_frac: float = 2.0e-5     # of image area
    var_threshold: float = 24.0
    learning_rate: float = 0.002       # slow: a briefly stopped vehicle is not absorbed into the background
    iou_gate: float = 0.05
    min_hits: int = 3
    max_coast: int = 12
    merge_iou: float = 0.08
    border_px: int = 2
    ghost_test: bool = False       # Cucchiara-style ghost suppression; off: conflicts with refined backgrounds


def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class _Track:
    _next_id = 0

    def __init__(self, bbox, frame_idx):
        self.id = _Track._next_id
        _Track._next_id += 1
        x, y, w, h = bbox
        kf = cv2.KalmanFilter(6, 4)
        # state: cx, cy, w, h, vcx, vcy
        kf.transitionMatrix = np.eye(6, dtype=np.float32)
        kf.transitionMatrix[0, 4] = 1.0
        kf.transitionMatrix[1, 5] = 1.0
        kf.measurementMatrix = np.zeros((4, 6), np.float32)
        kf.measurementMatrix[:4, :4] = np.eye(4, dtype=np.float32)
        kf.processNoiseCov = np.diag([1, 1, 1, 1, 4, 4]).astype(np.float32)
        kf.measurementNoiseCov = np.diag([2, 2, 4, 4]).astype(np.float32)
        kf.errorCovPost = np.diag([10, 10, 10, 10, 100, 100]).astype(np.float32)
        kf.statePost = np.array([[x + w / 2], [y + h / 2], [w], [h], [0], [0]], np.float32)
        self.kf = kf
        self.hits = 1
        self.coast = 0
        self.occluded = False
        x0, y0, w0, h0 = bbox
        self.records = [{"frame": frame_idx, "bbox": [float(v) for v in bbox], "status": "MEASURED",
                         "contact_uv": [x0 + w0 / 2.0, y0 + h0]}]

    def predict(self):
        s = self.kf.predict().ravel()
        cx, cy, w, h = s[:4]
        return [float(cx - w / 2), float(cy - h / 2), float(max(w, 1)), float(max(h, 1))]

    def update(self, bbox, frame_idx, contact_uv=None, status="MEASURED", correct_box=True, contact_span=None):
        x, y, w, h = bbox
        if correct_box:
            self.kf.correct(np.array([[x + w / 2], [y + h / 2], [w], [h]], np.float32))
        self.hits += 1
        self.coast = 0
        self.records.append({"frame": frame_idx, "bbox": [float(v) for v in bbox], "status": status,
                             "contact_uv": contact_uv or [x + w / 2.0, y + h], "contact_span": contact_span})

    def mark(self, frame_idx, status, bbox=None):
        self.coast += 1
        rec = {"frame": frame_idx, "bbox": bbox, "status": status}
        self.records.append(rec)


def _grad_mag(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def detect_foreground_blobs(sub, frame, cfg: TrackerConfig, learning_rate, bg_grad=None):
    h, w = frame.shape[:2]
    fg = sub.apply(frame, learningRate=learning_rate)
    fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]          # MOG2 marks shadows as 127
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg_raw = fg.copy()
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    min_area = cfg.min_area_frac * h * w
    cur_grad = _grad_mag(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if bg_grad is not None else None
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        if bg_grad is not None:
            # ghost test (Cucchiara et al. 2003): a real object has edges along its outline in the CURRENT
            # frame; a ghost (an object that is in the background model but not in the scene) has them only
            # in the background image.
            comp = (labels[y: y + bh, x: x + bw] == i).astype(np.uint8)
            edge = (comp - cv2.erode(comp, np.ones((3, 3), np.uint8))) > 0
            if edge.sum() >= 8:
                gc = float(cur_grad[y: y + bh, x: x + bw][edge].mean())
                gb = float(bg_grad[y: y + bh, x: x + bw][edge].mean())
                if gb > 12.0 and gc < 0.6 * gb:
                    continue
        # ground contact = the lowest row where the component is SOLID (at least half its typical row
        # width). Thin appendages -- compression ringing along a paint line, a shadow sliver -- are ignored.
        # For a fused blob these rows belong to the NEARER vehicle, which occludes the other.
        comp = (labels[y: y + bh, x: x + bw] == i) & (fg_raw[y: y + bh, x: x + bw] > 0)   # widths before closing
        row_w = comp.sum(axis=1)
        med_w = np.median(row_w[row_w > 0]) if np.any(row_w > 0) else 0
        solid = np.where(row_w >= 0.5 * med_w)[0]
        r_bot = int(solid.max()) if len(solid) else bh - 1
        band = comp[max(0, r_bot - 2): r_bot + 1]
        cols = np.where(band.any(axis=0))[0]
        if len(cols) > 2:
            # keep the dominant contiguous run of columns (drop a stray line pixel beside the body)
            runs = np.split(cols, np.where(np.diff(cols) > 2)[0] + 1)
            cols = max(runs, key=len)
        cu = float(x + cols.mean()) if len(cols) else float(x + bw / 2.0)
        span = [float(x + cols.min()), float(x + cols.max() + 1)] if len(cols) else [float(x), float(x + bw)]
        bh_eff = r_bot + 1
        out.append({"bbox": [float(x), float(y), float(bw), float(bh_eff)], "area": int(area),
                    "contact_uv": [cu, float(y + bh_eff)], "contact_span": span,
                    "touches_border": bool(x <= cfg.border_px or y <= cfg.border_px or
                                           x + bw >= w - cfg.border_px or y + bh_eff >= h - cfg.border_px)})
    return out


def _truncated_by(det, others, tol_px=4.0, min_h_overlap=0.3, tol_rel=0.3):
    """True if det's bottom edge coincides with the top edge of a box below it
    (an occluding, nearer vehicle) with enough horizontal overlap."""
    dx, dy, dw, dh = det
    bottom = dy + dh
    for ox, oy, ow, oh in others:
        if oy + oh <= bottom:          # the other box must extend below (be nearer)
            continue
        hov = max(0.0, min(dx + dw, ox + ow) - max(dx, ox)) / max(dw, 1.0)
        if hov >= min_h_overlap and abs(bottom - oy) <= max(tol_px, tol_rel * oh):
            return True
    return False


def track_vehicles(frames, background, cfg: TrackerConfig = None):
    """Returns list of tracks: {id, records:[{frame, bbox, status}], ...}.
    status in MEASURED | COASTED | MERGED | BORDER."""
    cfg = cfg or TrackerConfig()
    _Track._next_id = 0
    sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=cfg.var_threshold, detectShadows=True)
    for _ in range(30):
        sub.apply(background, learningRate=-1)
    bg_grad = _grad_mag(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)) if cfg.ghost_test else None
    tracks: List[_Track] = []
    finished: List[_Track] = []
    for fi, frame in enumerate(frames):
        dets = detect_foreground_blobs(sub, frame, cfg, cfg.learning_rate, bg_grad)
        preds = [t.predict() for t in tracks]
        # merged detections: substantially cover the predictions of >= 2 tracks
        def cover(pred, det):
            px, py, pw, ph = pred
            dx, dy, dw, dh = det
            ix = max(0.0, min(px + pw, dx + dw) - max(px, dx))
            iy = max(0.0, min(py + ph, dy + dh) - max(py, dy))
            return ix * iy / max(pw * ph, 1.0)
        merged_det = {}
        merged_tracks = set()
        for di, d in enumerate(dets):
            overl = [ti for ti, p in enumerate(preds) if cover(p, d["bbox"]) > 0.3 and tracks[ti].hits >= cfg.min_hits]
            if len(overl) >= 2:
                merged_det[di] = overl
                merged_tracks.update(overl)
        handled_t = set()
        for di, overl in merged_det.items():
            d = dets[di]
            # nearest vehicle = the one whose predicted bottom edge is lowest in the image
            near = max(overl, key=lambda ti: preds[ti][1] + preds[ti][3])
            bottom_ok = abs((preds[near][1] + preds[near][3]) - (d["bbox"][1] + d["bbox"][3])) < max(6.0, 0.25 * preds[near][3])
            if bottom_ok and not d["touches_border"]:
                # bottom edge still belongs to the nearer vehicle: measure its contact, but do not let the
                # fused box corrupt its size/centre state
                x, y, bw, bh = preds[near]
                tracks[near].update([x, d["bbox"][1] + d["bbox"][3] - bh, bw, bh], fi, d["contact_uv"],
                                    status="MEASURED_OCCLUDING", correct_box=False)
                tracks[near].kf.statePost[1, 0] = d["bbox"][1] + d["bbox"][3] - bh / 2.0
            else:
                tracks[near].mark(fi, "MERGED", preds[near])
            handled_t.add(near)
            for ti in overl:
                if ti != near:
                    tracks[ti].mark(fi, "OCCLUDED", preds[ti])
                    tracks[ti].coast = min(tracks[ti].coast, 1)   # occluded, not lost: keep alive
                    handled_t.add(ti)
        cand_d = [i for i in range(len(dets)) if i not in merged_det]
        matched_t, matched_d = set(), set()
        assigned_boxes = []

        def assign(track_ids, det_ids, gate):
            if not track_ids or not det_ids:
                return
            cost = np.ones((len(track_ids), len(det_ids)))
            for a, ti in enumerate(track_ids):
                for b, di in enumerate(det_ids):
                    cost[a, b] = 1.0 - _iou(preds[ti], dets[di]["bbox"])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= 1.0 - gate:
                    ti, di = track_ids[r], det_ids[c]
                    d = dets[di]
                    if d["touches_border"]:
                        tracks[ti].mark(fi, "BORDER", d["bbox"])
                    elif (d["bbox"][3] < 0.75 * preds[ti][3] and
                          _truncated_by(d["bbox"], assigned_boxes + [preds[q] for q in range(len(tracks)) if q != ti and not getattr(tracks[q], "occluded", False)])):
                        # bottom edge sits on a nearer vehicle's roofline: not a ground contact
                        tracks[ti].mark(fi, "OCCLUDED", d["bbox"])
                        tracks[ti].occluded = True
                    else:
                        tracks[ti].update(d["bbox"], fi, d["contact_uv"], contact_span=d.get("contact_span"))
                        tracks[ti].occluded = False
                    matched_t.add(ti)
                    matched_d.add(di)
                    assigned_boxes.append(d["bbox"])

        visible = [i for i in range(len(tracks)) if i not in handled_t and not getattr(tracks[i], "occluded", False)]
        hidden = [i for i in range(len(tracks)) if i not in handled_t and getattr(tracks[i], "occluded", False)]
        assign(visible, cand_d, cfg.iou_gate)
        # an occluded vehicle can only be re-acquired by a blob that is clearly separate from the
        # blobs just given to visible tracks (otherwise it is the occluder's blob)
        leftovers = [di for di in cand_d if di not in matched_d and
                     not any(cover(ab, dets[di]["bbox"]) > 0.1 or cover(dets[di]["bbox"], ab) > 0.1 for ab in assigned_boxes)]
        assign(hidden, leftovers, 0.2)
        for ti in handled_t:
            for di2, overl in merged_det.items():
                if ti in overl and ti != max(overl, key=lambda q: preds[q][1] + preds[q][3]):
                    tracks[ti].occluded = True
        for ti, t in enumerate(tracks):
            if ti in matched_t or ti in handled_t:
                continue
            if getattr(t, "occluded", False):
                t.mark(fi, "OCCLUDED", preds[ti])
                t.coast = min(t.coast, cfg.max_coast // 2)   # stays alive while hidden behind its occluder
            else:
                t.mark(fi, "COASTED", preds[ti])
        occluded_boxes = [preds[ti] for ti in handled_t]
        for di in cand_d:
            if di in matched_d or dets[di]["touches_border"]:
                continue
            if any(cover(ob, dets[di]["bbox"]) > 0.2 or _iou(ob, dets[di]["bbox"]) > 0.05 for ob in occluded_boxes):
                continue   # part of a fused blob -- not a new vehicle
            tracks.append(_Track(dets[di]["bbox"], fi))
        alive = []
        for t in tracks:
            if t.coast > cfg.max_coast:
                finished.append(t)
            else:
                alive.append(t)
        tracks = alive
    finished.extend(tracks)
    out = []
    for t in finished:
        n_meas = sum(1 for r in t.records if r["status"] in ("MEASURED", "MEASURED_OCCLUDING"))
        if n_meas < cfg.min_hits:
            continue
        # trim trailing non-measured records
        recs = list(t.records)
        while recs and recs[-1]["status"] not in ("MEASURED", "MEASURED_OCCLUDING", "MERGED", "OCCLUDED"):
            recs.pop()
        out.append({"id": t.id, "records": recs, "n_measured": n_meas})
    out.sort(key=lambda tr: tr["records"][0]["frame"])
    return out


def bottom_centre(bbox):
    x, y, w, h = bbox
    return (x + w / 2.0, y + h)


def contact_point(record):
    return tuple(record.get("contact_uv") or bottom_centre(record["bbox"]))


def refine_background(frames, tracks, base_bg, n=300, dilate_px=6, min_samples=3, border_px=4, use_first_half_where_changed=True):
    """Second-pass background: per-pixel temporal median over sampled frames,
    EXCLUDING pixels covered by a tracked vehicle in that frame. A track that
    vanishes away from the image border is assumed to have stopped (or to be
    hidden) where it was last seen, so its last box stays excluded until the
    end of the clip -- this keeps vehicles that come to rest after a
    collision out of the background. Pixels with too few clean samples keep
    the base estimate."""
    nF = len(frames)
    h, w = frames[0].shape[:2]
    idx = np.linspace(0, nF - 1, min(n, nF)).astype(int)
    masks = np.zeros((len(idx), h, w), bool)
    by_frame = {}
    for tr in tracks:
        recs = [r for r in tr["records"] if r.get("bbox")]
        for r in recs:
            by_frame.setdefault(r["frame"], []).append(r["bbox"])
        meas = [r for r in tr["records"] if r.get("bbox") and r["status"] in ("MEASURED", "MEASURED_OCCLUDING")]
        if meas:
            last = meas[-1]      # last place the vehicle was actually SEEN (not where a coasting filter drifted to)
            x, y, bw, bh = last["bbox"]
            at_border = x <= border_px or y <= border_px or x + bw >= w - border_px or y + bh >= h - border_px
            if not at_border:
                for f in range(last["frame"] + 1, nF):
                    by_frame.setdefault(f, []).append(last["bbox"])
    for k, f in enumerate(idx):
        for (x, y, bw, bh) in by_frame.get(int(f), []):
            x0, y0 = max(0, int(x) - dilate_px), max(0, int(y) - dilate_px)
            x1, y1 = min(w, int(x + bw) + dilate_px), min(h, int(y + bh) + dilate_px)
            masks[k, y0:y1, x0:x1] = True
    touched = masks.any(axis=0)
    out = base_bg.copy()
    if not touched.any():
        return out, 0
    ys, xs = np.where(touched)
    stack = np.stack([frames[i][ys, xs] for i in idx]).astype(np.float32)   # (n, P, 3)
    m = masks[:, ys, xs]
    stack[m] = np.nan
    good = (~m).sum(axis=0) >= min_samples
    med = np.nanmedian(stack[:, good], axis=0)
    out[ys[good], xs[good]] = np.clip(med, 0, 255).astype(np.uint8)
    if use_first_half_where_changed and len(idx) >= 20:
        # where the first- and second-half medians disagree, something came to rest (or left). In
        # event footage vehicles come to rest AFTER the event, so the earlier appearance is the road.
        half = len(idx) // 2
        b1 = np.median(np.stack([frames[i] for i in idx[:half]]), axis=0)
        b2 = np.median(np.stack([frames[i] for i in idx[half:]]), axis=0)
        changed = np.abs(b1 - b2).max(axis=2) > 30
        changed = cv2.dilate(changed.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        if changed.any():
            cy, cx = np.where(changed)
            st = np.stack([frames[i][cy, cx] for i in idx[:half]]).astype(np.float32)
            mm = masks[:half][:, cy, cx]
            st[mm] = np.nan
            ok = (~mm).sum(axis=0) >= min_samples
            if ok.any():
                out[cy[ok], cx[ok]] = np.clip(np.nanmedian(st[:, ok], axis=0), 0, 255).astype(np.uint8)
    return out, int(touched.sum())


def stable_background(frames, fps, min_run_s=1.0, diff_thresh=10.0, fallback=None):
    """Per-pixel background = mean of the EARLIEST run of >= min_run_s frames
    in which the pixel does not change (frame-to-frame max-channel difference
    below diff_thresh). Rationale for event footage: the road surface where
    vehicles will end up (after braking or a collision) is visible, unchanging,
    BEFORE they arrive; a whole-clip median would instead learn the stopped
    vehicles. Pixels without such a run fall back to the temporal median.
    Limitation (stated): a vehicle that never moves during the clip is
    indistinguishable from the road and is never detected."""
    T = len(frames)
    h, w = frames[0].shape[:2]
    need = max(5, int(round(min_run_s * fps)))
    run = np.zeros((h, w), np.int32)
    acc = np.zeros((h, w, 3), np.float32)
    found = np.zeros((h, w), bool)
    out = np.zeros((h, w, 3), np.float32)
    prev = frames[0].astype(np.int16)
    acc += prev
    run[:] = 1
    for t in range(1, T):
        cur = frames[t].astype(np.int16)
        stable = (np.abs(cur - prev).max(axis=2) < diff_thresh) & ~found
        # continue runs where stable, restart elsewhere
        run = np.where(stable, run + 1, 1)
        acc = np.where(stable[..., None], acc + cur, cur.astype(np.float32))
        done = stable & (run >= need)
        if done.any():
            out[done] = acc[done] / run[done][:, None]
            found |= done
        prev = cur
        if found.all():
            break
    if fallback is None:
        fallback = np.median(np.stack(frames[:: max(1, T // 48)]), axis=0)
    out[~found] = fallback[~found]
    return np.clip(out, 0, 255).astype(np.uint8), float(found.mean())
