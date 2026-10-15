"""
Ground-truth-free event detection for multi-camera sync
(docs/multi_camera_architecture.md, scoped next-step #1).

The proof-of-concept version of this (run_multi_camera_cross_examination.py's
first draft) used the synthetic ground truth's `collision_frame` as the
shared sync anchor -- a real simplification, explicitly flagged as
needing replacement before this works on any real footage. This module
is that replacement: detects the closest-approach moment between two
independently-tracked vehicles using ONLY each camera's own
reconstruction, no ground truth.

Approach: track the full clip first (no truncation -- a real system
doesn't know in advance where the interesting part ends), find the
local time of closest approach between the two tracked vehicles'
independently-reconstructed positions (interpolated onto a shared time
grid since the two tracks may have different frame-loss patterns), then
re-run tracking a second time truncated just past that point. This
mirrors the deliberate design already used with ground truth in
run_causality_feasibility.py's run_collision_pipeline (truncating before
post-collision merged-blob frames contaminate the trajectory smoother)
-- the truncation logic isn't new, only how the cutoff point is found.

This detects CLOSEST APPROACH, not specifically "collision" -- for a
scenario that doesn't collide, this still finds a well-defined,
meaningful event (nearest point between the two vehicles), which is
exactly what's needed as a sync anchor regardless of whether a collision
actually occurs.
"""
import numpy as np


def detect_closest_approach(track_a, track_b, min_overlap_s=0.3):
    """
    track_a/track_b: dicts with "t_s" and "x_m" arrays (the shape
    produced by run_causality_feasibility.py's track_vehicle /
    track_two_vehicles_bgsub).

    Returns {"t_local_s": float, "gap_m": float, "frame_idx_estimate": int}
    or None if the two tracks don't share enough overlapping time to
    compare (e.g. one vehicle was never visible while the other was).
    """
    if len(track_a.get("t_s", [])) < 2 or len(track_b.get("t_s", [])) < 2:
        return None
    t_lo = max(track_a["t_s"][0], track_b["t_s"][0])
    t_hi = min(track_a["t_s"][-1], track_b["t_s"][-1])
    if t_hi - t_lo < min_overlap_s:
        return None

    n_samples = max(20, int((t_hi - t_lo) * 60))
    t_common = np.linspace(t_lo, t_hi, n_samples)
    xa = np.interp(t_common, track_a["t_s"], track_a["x_m"])
    xb = np.interp(t_common, track_b["t_s"], track_b["x_m"])
    gap = np.abs(xa - xb)
    # NOT argmin(gap): a real collision freezes the vehicles touching for
    # the rest of the clip, so the gap plateaus near its minimum for a
    # long tail afterward rather than having one sharp minimum -- a
    # global argmin picks an arbitrary point in that plateau (confirmed
    # by inspection: it was matching the WRONG camera's true event time
    # almost exactly, a near-swap that was actually two independent
    # plateau-sampling artifacts, not a real correspondence). Take the
    # FIRST time the gap reaches within a small tolerance of its eventual
    # minimum instead -- the first-contact instant, not an arbitrary
    # later point on the plateau.
    min_gap = gap.min()
    idx = int(np.argmax(gap <= min_gap + 0.5))
    t_event = float(t_common[idx])

    # rough frame index: whichever track has denser sampling near the
    # event determines the natural resolution to truncate at
    dt_a = np.median(np.diff(track_a["t_s"])) if len(track_a["t_s"]) > 1 else 1.0 / 30
    frame_idx_estimate = int(round(t_event / dt_a))
    return {"t_local_s": t_event, "gap_m": float(gap[idx]), "frame_idx_estimate": frame_idx_estimate}


def reconstruct_with_self_discovered_truncation(frames, H, fps, tracker_fn, margin_frames=3):
    """
    tracker_fn(frames, H, fps, frame_limit) -> (track_a, track_b), i.e.
    track_two_vehicles_bgsub's signature -- passed in rather than
    imported directly so this module has no dependency on the
    evaluation scripts (keeps multiview reusable from production code,
    not just eval harnesses).

    Two-pass: first pass tracks the whole clip to find the event via
    detect_closest_approach, second pass re-tracks truncated just past
    that point (mirrors the ground-truth-driven truncation in
    run_collision_pipeline, but self-discovered).

    Returns (track_a, track_b, event) where event is
    detect_closest_approach's result (or None if no event was found, in
    which case the untruncated full-clip tracks are returned as-is).
    """
    track_a_full, track_b_full = tracker_fn(frames, H, fps, None)
    event = detect_closest_approach(track_a_full, track_b_full)
    if event is None:
        return track_a_full, track_b_full, None

    frame_limit = min(len(frames), event["frame_idx_estimate"] + margin_frames)
    track_a, track_b = tracker_fn(frames, H, fps, frame_limit)
    return track_a, track_b, event
