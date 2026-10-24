"""
Annotated evidence video (brief section 22 + section 27's demo script):
the primary demo artifact. Everything drawn corresponds to a real number
already present in the pipeline's result dict -- lane-edge lines
actually fit by cv/lines, vehicle boxes actually returned by the
tracker, speeds/gap/TTC from the actual reconstruction, and a
"required stopping point" marker computed from the actual avoidability
physics and projected back onto the road via the actual fitted
homography (not a hand-placed illustration).

Visual language matches the web UI (HUD-teal for measurement/tracking,
amber/hazard-red for the avoidability verdict, monospace HUD text) so
the video and the dashboard read as the same product.
"""
import cv2
import numpy as np

from backend.causality.avoidability import required_stopping_distance

# BGR, matching the frontend's --hud/--signal/--hazard/--paper/--muted tokens
HUD_TEAL = (198, 209, 127)
SIGNAL_AMBER = (61, 163, 232)
HAZARD_RED = (60, 68, 193)
PAPER = (226, 234, 237)
MUTED = (150, 146, 139)
PANEL_BG = (14, 13, 11)

VERDICT_COLOR = {
    "AVOIDABLE": SIGNAL_AMBER,
    "NOT_AVOIDABLE": HAZARD_RED,
    "INSUFFICIENT_EVIDENCE": MUTED,
}

MPS_TO_MPH = 2.23694
M_TO_FT = 3.28084


def _speed_str(mps, units):
    if units == "imperial":
        return f"{mps * MPS_TO_MPH:5.1f} mph"
    return f"{mps:5.1f} m/s"


def _dist_str(m, units):
    if units == "imperial":
        return f"{m * M_TO_FT:.0f} ft"
    return f"{m:.0f} m"


def _panel(frame, x0, y0, x1, y1, alpha=0.62, border=None):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    if border is not None:
        cv2.rectangle(frame, (x0, y0), (x1, y1), border, 1, cv2.LINE_AA)


def _text(frame, text, org, color=PAPER, scale=0.5, thickness=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_line_x_of_y(frame, line, color, thickness=1):
    if line is None:
        return
    m, b = line
    h = frame.shape[0]
    pts = np.array([(m * v + b, v) for v in range(0, h, 4)], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], False, color, thickness, cv2.LINE_AA)


def _car_polygon(x0, y0, x1, y1):
    """
    Approximates a vehicle body silhouette from a plain axis-aligned
    bounding box -- the tracker only returns (x, y, w, h), no heading,
    so this can't be a true oriented car outline, but a beveled octagon
    (narrower "roof" on top, full-width body with cut corners below)
    reads as a vehicle rather than a generic debug rectangle, which a
    plain rounded rect never did. Standard technique for exactly this
    bbox-only-data situation.
    """
    w, h = x1 - x0, y1 - y0
    roof_inset = max(2, int(w * 0.16))
    bevel = max(2, int(h * 0.22))
    return np.array([
        [x0 + roof_inset, y0],           # roof, left
        [x1 - roof_inset, y0],           # roof, right
        [x1, y0 + bevel],                # shoulder, right
        [x1, y1 - bevel],                # body, right
        [x1 - bevel, y1],                # bumper, right
        [x0 + bevel, y1],                # bumper, left
        [x0, y1 - bevel],                # body, left
        [x0, y0 + bevel],                # shoulder, left
    ], dtype=np.int32)


def _glow_rounded_box(frame, bbox, color, pad=5, frame_idx=0, glow_layers=3):
    """
    Futuristic vehicle lock-on marker: a beveled car-silhouette outline
    (see _car_polygon) rather than a plain rectangle, with a soft outer
    glow, a faint tracking fill, angular corner accents at the body
    shoulders, and a scanline sweep that actually moves frame to frame
    (this is a video, not a static image -- real per-frame motion is
    free "wow factor" a static UI screenshot can't offer) so it reads
    as an active sensor lock rather than a debug overlay.
    """
    x, y, w, h = bbox
    x0, y0, x1, y1 = int(x - pad), int(y - pad), int(x + w + pad), int(y + h + pad)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame.shape[1] - 1, x1), min(frame.shape[0] - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return

    poly = _car_polygon(x0, y0, x1, y1)

    # faint tracking fill, so the locked vehicle reads as a solid tracked
    # object, not just a wireframe
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, dst=frame)

    # soft outer glow -- concentric expanded outlines at decreasing alpha
    overlay = frame.copy()
    for i in range(glow_layers, 0, -1):
        expand = i * 2
        expanded = _car_polygon(x0 - expand, y0 - expand, x1 + expand, y1 + expand)
        cv2.polylines(overlay, [expanded], True, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, dst=frame)

    # crisp body outline
    cv2.polylines(frame, [poly], True, color, 1, cv2.LINE_AA)

    # animated scanline: a single bright horizontal band that sweeps
    # top-to-bottom across the silhouette, position keyed by frame_idx --
    # genuinely moves across the rendered video, not a static decoration
    if y1 > y0:
        cycle = max(8, int((y1 - y0) * 1.4))
        sweep_y = y0 + (frame_idx * 2) % cycle
        if y0 <= sweep_y <= y1:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            band = frame.copy()
            cv2.line(band, (x0, sweep_y), (x1, sweep_y), color, 2, cv2.LINE_AA)
            band_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.line(band_mask, (x0, sweep_y), (x1, sweep_y), 255, 2)
            band_mask = cv2.bitwise_and(band_mask, mask)
            frame[band_mask > 0] = cv2.addWeighted(frame, 0.35, band, 0.65, 0)[band_mask > 0]

    # angular corner accents at the body shoulders (echoes the UI's
    # reticle-corner motif) -- short diagonal ticks, not full brackets
    tick = 7
    for cx, cy, dx, dy in [(x0, y0 + max(2, int((y1 - y0) * 0.22)), 1, -1),
                            (x1, y0 + max(2, int((y1 - y0) * 0.22)), -1, -1),
                            (x0, y1 - max(2, int((y1 - y0) * 0.22)), 1, 1),
                            (x1, y1 - max(2, int((y1 - y0) * 0.22)), -1, 1)]:
        cv2.line(frame, (cx, cy), (cx + dx * tick, cy + dy * tick), color, 2, cv2.LINE_AA)

    return x0, y0, x1, y1


def _trend_arrow(cur, prev, eps=0.15):
    """Real, not decorative: derived from the actual frame-to-frame
    speed delta already in the pipeline's reconstruction."""
    if prev is None:
        return None
    d = cur - prev
    if d > eps:
        return "up"
    if d < -eps:
        return "down"
    return None


def _telemetry_chip(frame, vehicle_id, speed_str, trend, anchor_xy, color, above=True):
    """
    Compact HUD telemetry chip replacing the old plain text label: a
    filled ID badge, the speed reading, and a trend glyph (accel/decel)
    computed from the real frame-to-frame speed delta.
    """
    (tw, th), _ = cv2.getTextSize(speed_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    badge_w, trend_w = 15, (10 if trend else 0)
    x, y = anchor_xy
    total_w = badge_w + tw + trend_w + 16
    x0 = int(x - total_w / 2)
    y1 = y - 7 if above else y + 7
    y0 = y1 - th - 11 if above else y1
    y1b = y1 if above else y1 + th + 11
    x1 = x0 + total_w
    _panel(frame, x0, y0, x1, y1b, alpha=0.7, border=color)
    cv2.rectangle(frame, (x0, y0), (x0 + badge_w, y1b), color, -1)
    text_y = y1b - 6 if above else y0 + th + 2
    cv2.putText(frame, vehicle_id, (x0 + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, PANEL_BG, 1, cv2.LINE_AA)
    _text(frame, speed_str, (x0 + badge_w + 5, text_y), PAPER, 0.42)
    if trend:
        tx, tymid = x1 - 9, (y0 + y1b) // 2
        if trend == "up":
            pts = np.array([[tx - 4, tymid + 3], [tx + 4, tymid + 3], [tx, tymid - 4]], np.int32)
            cv2.fillConvexPoly(frame, pts, SIGNAL_AMBER)
        else:
            pts = np.array([[tx - 4, tymid - 3], [tx + 4, tymid - 3], [tx, tymid + 4]], np.int32)
            cv2.fillConvexPoly(frame, pts, HUD_TEAL)
    return x0, y0, x1, y1b


def _heading_chevron(frame, trail, box, color, length=14):
    """Small directional chevron just ahead of the vehicle silhouette,
    pointing along its actual recent direction of travel (derived from
    the last two tracked image-space positions, not assumed)."""
    if len(trail) < 2 or box is None:
        return
    (px, py), (cx, cy) = trail[-2], trail[-1]
    dx, dy = cx - px, cy - py
    norm = (dx ** 2 + dy ** 2) ** 0.5
    if norm < 1e-3:
        return
    dx, dy = dx / norm, dy / norm
    x0, y0, x1, y1 = box
    tip_x, tip_y = int(cx + dx * length), int(cy + dy * length)
    perp_x, perp_y = -dy, dx
    base1 = (int(tip_x - dx * 7 + perp_x * 5), int(tip_y - dy * 7 + perp_y * 5))
    base2 = (int(tip_x - dx * 7 - perp_x * 5), int(tip_y - dy * 7 - perp_y * 5))
    pts = np.array([base1, (tip_x, tip_y), base2], np.int32)
    cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)


def _inline_label(frame, text, anchor_xy, color, above=True):
    """Small floating label anchored just above/below a vehicle's box,
    with its own background plate so it stays legible over any road color."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    x, y = anchor_xy
    x0 = int(x - tw / 2) - 4
    y1 = y - 6 if above else y + 6
    y0 = y1 - th - 6 if above else y1
    y1b = y1 if above else y1 + th + 6
    x1 = x0 + tw + 8
    _panel(frame, x0, y0, x1, y1b, alpha=0.6)
    _text(frame, text, (x0 + 4, y1b - 5 if above else y0 + th + 1), color, 0.4)


def _stopping_point_marker(H, left_line, right_line, world_x, image_h):
    if left_line is None or right_line is None:
        return None
    from cv.homography.ground_plane import image_to_world
    ml, bl = left_line
    mr, br = right_line
    v_lo, v_hi = 20, image_h - 5
    best_row, best_diff = None, None
    for v in range(v_lo, v_hi, 4):
        ul, ur = ml * v + bl, mr * v + br
        wx = image_to_world(H, [[(ul + ur) / 2.0, v]])[0][0]
        diff = abs(wx - world_x)
        if best_diff is None or diff < best_diff:
            best_diff, best_row = diff, v
    if best_row is None:
        return None
    near_x = image_to_world(H, [[(ml * v_hi + bl + mr * v_hi + br) / 2.0, v_hi]])[0][0]
    far_x = image_to_world(H, [[(ml * v_lo + bl + mr * v_lo + br) / 2.0, v_lo]])[0][0]
    off_frame_by_m = 0.0
    if world_x < near_x:
        off_frame_by_m = near_x - world_x
    elif world_x > far_x:
        off_frame_by_m = world_x - far_x
    ul, ur = ml * best_row + bl, mr * best_row + br
    return (int(ul), best_row), (int(ur), best_row), off_frame_by_m


def annotate_video(video_path, result, output_path, fps=None, units="metric"):
    """units: 'metric' (m/s, meters) or 'imperial' (mph, feet)."""
    cap = cv2.VideoCapture(video_path)
    src_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))

    geometry = result.get("geometry", {})
    left_line, right_line = geometry.get("left_line"), geometry.get("right_line")
    H = result.get("_H")

    track_a = result.get("_tracks", {}).get("vehicle_a", {})
    track_b = result.get("_tracks", {}).get("vehicle_b", {})
    uv_a = dict(zip(track_a.get("frame_idx", []), track_a.get("uv", [])))
    uv_b = dict(zip(track_b.get("frame_idx", []), track_b.get("uv", [])))
    bbox_a = dict(zip(track_a.get("frame_idx", []), track_a.get("bbox", [])))
    bbox_b = dict(zip(track_b.get("frame_idx", []), track_b.get("bbox", [])))

    traj = result.get("trajectories", {})
    speed_a = dict(zip(track_a.get("frame_idx", []), traj.get("vehicle_a", {}).get("speed_mps", [])))
    speed_b = dict(zip(track_b.get("frame_idx", []), traj.get("vehicle_b", {}).get("speed_mps", [])))
    t_by_frame_b = dict(zip(track_b.get("frame_idx", []), traj.get("vehicle_b", {}).get("t_s", [])))

    causality = result.get("causality", {})
    verdict = causality.get("verdict", "")
    verdict_color = VERDICT_COLOR.get(verdict, PAPER)
    collision_time_s = result.get("collision_time_s")
    assumptions = causality.get("assumptions", {})
    max_decel = assumptions.get("max_decel_mps2")
    reaction_time = assumptions.get("reaction_time_s")

    uncertainty = result.get("uncertainty", {})
    skeptic_findings = geometry.get("skeptic_findings")
    skeptic_disputed = geometry.get("skeptic_disputed")

    braking_onset_before = causality.get("braking_onset_time_before_impact_s")
    braking_onset_t = (collision_time_s - braking_onset_before) if (braking_onset_before and collision_time_s) else None
    steering_onset_before = causality.get("steering_onset_time_before_impact_s")
    steering_onset_t = (collision_time_s - steering_onset_before) if (steering_onset_before and collision_time_s) else None
    steering_direction = causality.get("steering_direction")

    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stop_marker_world_x = None
    v0 = causality.get("striking_vehicle_speed_at_earliest_observed_mps")
    if v0 and max_decel and reaction_time and len(track_a.get("x_m", [])) > 0:
        needed = required_stopping_distance(v0, max_decel, reaction_time)
        collision_x = track_a["x_m"][-1] if len(track_a.get("x_m", [])) else None
        if collision_x is not None:
            stop_marker_world_x = collision_x - needed

    unit_suffix = "mph" if units == "imperial" else "m/s"

    trail_a, trail_b = [], []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        _draw_line_x_of_y(frame, left_line, (90, 90, 90), 1)
        _draw_line_x_of_y(frame, right_line, (90, 90, 90), 1)

        stop_off_frame_m = None
        if H is not None and stop_marker_world_x is not None:
            marker = _stopping_point_marker(H, left_line, right_line, stop_marker_world_x, h)
            if marker is not None:
                p1, p2, off_frame_m = marker
                if off_frame_m < 1.0:
                    cv2.line(frame, p1, p2, SIGNAL_AMBER, 2, cv2.LINE_AA)
                    _text(frame, "required stopping point (modeled)", (p1[0] - 40, p1[1] - 6), SIGNAL_AMBER, 0.38)
                else:
                    stop_off_frame_m = off_frame_m

        if frame_idx in uv_a:
            trail_a.append(uv_a[frame_idx])
        if frame_idx in uv_b:
            trail_b.append(uv_b[frame_idx])
        for trail, color in ((trail_a, MUTED), (trail_b, HUD_TEAL)):
            if len(trail) >= 2:
                pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

        # futuristic per-vehicle outline + telemetry chip + heading chevron
        if frame_idx in bbox_a:
            box = _glow_rounded_box(frame, bbox_a[frame_idx], MUTED, frame_idx=frame_idx)
            _heading_chevron(frame, trail_a, box, MUTED)
            if box is not None and frame_idx in speed_a:
                trend = _trend_arrow(speed_a[frame_idx], speed_a.get(frame_idx - 1))
                _telemetry_chip(frame, "A", _speed_str(speed_a[frame_idx], units),
                                 trend, (box[0] + (box[2] - box[0]) // 2, box[1]), MUTED, above=True)
        if frame_idx in bbox_b:
            box = _glow_rounded_box(frame, bbox_b[frame_idx], HUD_TEAL, frame_idx=frame_idx)
            _heading_chevron(frame, trail_b, box, HUD_TEAL)
            if box is not None and frame_idx in speed_b:
                trend = _trend_arrow(speed_b[frame_idx], speed_b.get(frame_idx - 1))
                _telemetry_chip(frame, "B", _speed_str(speed_b[frame_idx], units),
                                 trend, (box[0] + (box[2] - box[0]) // 2, box[1]), HUD_TEAL, above=True)

        # --- top-left telemetry sidebar (redesigned: accent rule + tighter rows) ---
        cur_t = t_by_frame_b.get(frame_idx)
        ttc = None
        if cur_t is not None and collision_time_s is not None:
            ttc = max(0.0, collision_time_s - cur_t)

        rows = []
        if stop_off_frame_m is not None:
            rows.append((f"required stop: {_dist_str(stop_off_frame_m, units)} before visible road", SIGNAL_AMBER))
        if frame_idx in speed_a:
            rows.append((f"vehicle_a   {_speed_str(speed_a[frame_idx], units)}", MUTED))
        if frame_idx in speed_b:
            rows.append((f"vehicle_b   {_speed_str(speed_b[frame_idx], units)}", HUD_TEAL))
        if ttc is not None:
            rows.append((f"time to collision  {ttc:4.1f} s", PAPER))
        ci = uncertainty.get("striking_vehicle_speed_ci90_mps")
        if ci and ci[0] is not None:
            ci_lo = ci[0] * (MPS_TO_MPH if units == "imperial" else 1)
            ci_hi = ci[1] * (MPS_TO_MPH if units == "imperial" else 1)
            rows.append((f"90% CI  {ci_lo:.1f}-{ci_hi:.1f} {unit_suffix}", MUTED))
        if skeptic_findings:
            skeptic_label = "SKEPTIC: DISPUTED" if skeptic_disputed else "SKEPTIC: CLEARED"
            rows.append((skeptic_label, HAZARD_RED if skeptic_disputed else HUD_TEAL))

        pad, row_h = 8, 18
        header = "Stradiva // EVIDENCE"
        header_w = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        sidebar_w = max([header_w] + [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0] for t, _ in rows]) + pad * 2 + 6
        header_h = 20
        sidebar_h = header_h + pad + row_h * len(rows) + pad
        _panel(frame, 8, 8, 8 + sidebar_w, 8 + sidebar_h, border=(45, 43, 40))
        # teal header strip, echoing the frontend's accent-rule Eyebrow motif
        cv2.rectangle(frame, (8, 8), (8 + sidebar_w, 8 + header_h), HUD_TEAL, -1)
        cv2.putText(frame, header, (8 + pad, 8 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PANEL_BG, 1, cv2.LINE_AA)
        cv2.line(frame, (8, 8 + header_h), (8, 8 + sidebar_h), HUD_TEAL, 2, cv2.LINE_AA)  # left accent rule
        for i, (t, c) in enumerate(rows):
            _text(frame, t, (8 + pad + 4, 8 + header_h + pad + 12 + i * row_h), c, 0.46)

        # --- event flash markers ---
        flash_window_s = 0.6
        if cur_t is not None:
            if braking_onset_t is not None and 0 <= cur_t - braking_onset_t < flash_window_s:
                _panel(frame, w // 2 - 90, 8, w // 2 + 90, 34, alpha=0.75)
                _text(frame, "BRAKING DETECTED", (w // 2 - 78, 27), SIGNAL_AMBER, 0.5, 1)
            elif steering_onset_t is not None and 0 <= cur_t - steering_onset_t < flash_window_s:
                _panel(frame, w // 2 - 110, 8, w // 2 + 110, 34, alpha=0.75)
                _text(frame, f"STEERING DETECTED ({steering_direction})", (w // 2 - 98, 27), SIGNAL_AMBER, 0.5, 1)

        # --- verdict banner ---
        if verdict:
            label = verdict.replace("_", " ")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            x0, y0 = w - tw - 24, 8
            _panel(frame, x0 - 6, y0 - 6, x0 + tw + 6, y0 + th + 10, alpha=0.6)
            _text(frame, label, (x0, y0 + th + 2), verdict_color, 0.6)

        # --- bottom timeline ---
        bar_y = h - 14
        cv2.line(frame, (16, bar_y), (w - 16, bar_y), (60, 58, 54), 2, cv2.LINE_AA)
        frac = frame_idx / max(1, n_frames_total - 1)
        px = int(16 + frac * (w - 32))
        cv2.circle(frame, (px, bar_y), 4, HUD_TEAL, -1, cv2.LINE_AA)
        for onset_t in (braking_onset_t, steering_onset_t):
            if onset_t is not None and collision_time_s:
                onset_frac = onset_t / (n_frames_total / src_fps)
                ox = int(16 + onset_frac * (w - 32))
                cv2.line(frame, (ox, bar_y - 5), (ox, bar_y + 5), SIGNAL_AMBER, 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path
