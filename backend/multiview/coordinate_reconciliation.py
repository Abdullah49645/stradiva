"""
Coordinate reconciliation between cameras (docs/multi_camera_architecture.md,
scoped next-step #2).

Each camera's homography recovers a geometrically correct LOCAL
road-aligned coordinate system -- right scale, right 1D structure --
but with no inherent shared origin or direction relative to another
camera's frame. Confirmed directly during the multi-camera proof of
concept: camera B's reconstructed x_m ran in the OPPOSITE direction
from camera A's for the same physical motion (a backward-facing camera
naturally produces this), which is why the original cross-examination
could only safely compare speed MAGNITUDE, not position or signed
velocity.

This module fixes that: given each camera's own independent trajectory
and its own independently-detected event time (from
backend/multiview/event_detection.py -- no ground truth), it determines
(1) whether camera B's coordinate axis runs the same or opposite
direction as camera A's, and (2) the offset needed so both cameras'
reconstructions describe the SAME physical positions at the SAME
(synced) moments.

The shared event (closest approach) is the anchor: at that instant,
both cameras are looking at the same physical position, so their
local-x readings AT that moment -- after fixing the direction -- must
coincide. That's one equation in one unknown (the offset); direction
is resolved separately from each camera's own velocity sign.
"""
import numpy as np


def _direction_sign(t_s, x_m):
    """+1 if x_m increases with time (this camera's local axis runs the
    same way as physical motion), -1 if it decreases. Uses a linear fit
    across the whole track rather than just endpoints, so it's not
    thrown off by a single noisy sample at either end."""
    if len(t_s) < 2:
        return 1
    slope = np.polyfit(t_s, x_m, 1)[0]
    return 1 if slope >= 0 else -1


def reconcile(track_a, track_b, event_a_t_local_s, event_b_t_local_s):
    """
    track_a/track_b: dicts with "t_s" and "x_m" arrays, in each camera's
    OWN local time and local coordinate system.
    event_a_t_local_s / event_b_t_local_s: each camera's own
    independently-detected shared-event local time (e.g. from
    event_detection.detect_closest_approach on that camera's own
    tracks) -- NOT ground truth.

    Returns {
      "direction_sign": +1 or -1 (applied to camera B's x_m to align
                         its axis direction with camera A's),
      "offset_m": additive offset so reconciled camera B matches camera
                  A's coordinate frame at the shared event,
      "track_b_reconciled_x_m": camera B's track_b["x_m"], transformed
                  into camera A's coordinate frame,
    } or None if either track doesn't bracket its own event time (can't
    anchor the offset).
    """
    dir_a = _direction_sign(track_a["t_s"], track_a["x_m"])
    dir_b = _direction_sign(track_b["t_s"], track_b["x_m"])
    sign = 1 if dir_a == dir_b else -1

    if not (track_a["t_s"][0] <= event_a_t_local_s <= track_a["t_s"][-1]):
        return None
    if not (track_b["t_s"][0] <= event_b_t_local_s <= track_b["t_s"][-1]):
        return None

    x_a_at_event = float(np.interp(event_a_t_local_s, track_a["t_s"], track_a["x_m"]))
    x_b_at_event_raw = float(np.interp(event_b_t_local_s, track_b["t_s"], track_b["x_m"]))
    x_b_at_event_flipped = sign * x_b_at_event_raw
    offset = x_a_at_event - x_b_at_event_flipped

    track_b_reconciled = sign * np.asarray(track_b["x_m"]) + offset
    return {
        "direction_sign": sign,
        "offset_m": offset,
        "track_b_reconciled_x_m": track_b_reconciled,
        "x_a_at_event_m": x_a_at_event,
        "x_b_at_event_reconciled_m": sign * x_b_at_event_raw + offset,  # should equal x_a_at_event by construction
    }
