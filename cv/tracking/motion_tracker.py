"""
Motion-based vehicle detection (replaces the Week 1/2 color-blob
placeholder flagged throughout as synthetic-only). Uses MOG2 background
subtraction against a static-camera assumption -- appropriate for
dashcam-mounted-and-stationary, CCTV, and fixed traffic-camera footage,
which covers most of the accident-footage sources in scope (section 1).
Does NOT depend on vehicle color, unlike the earlier placeholder.

Multi-vehicle disambiguation for the two-vehicle collision case uses the
same geometric fact exploited elsewhere in this codebase: for this
camera model (zero roll), image row v decreases monotonically with
world forward-distance X, so the farther (lead) vehicle has a smaller v
than the nearer (following) vehicle, as long as their depth order
doesn't swap -- true for the rear-end scenarios this module targets.
A real deployment would need proper multi-object tracking (Kalman
filter + Hungarian assignment, section 13) instead of this ordering
heuristic; that upgrade is still open (see docs).
"""
import cv2
import numpy as np


class BackgroundSubtractionTracker:
    def __init__(self, history=500, var_threshold=32, detect_shadows=True,
                 min_area=25, learning_rate=-1):
        self.sub = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
        self.min_area = min_area
        self.learning_rate = learning_rate  # -1 = auto
        self.kernel = np.ones((3, 3), np.uint8)

    def seed_background(self, clean_frame, n_seed_iters=30):
        """
        Prime the background model from a known-clean (vehicle-free)
        reference frame instead of learning it online from the first N
        frames of the clip. Necessary whenever a vehicle is already
        in-frame at t=0 (e.g. a lead vehicle stopped before the clip
        starts) -- online warmup alone can absorb it into the background
        before any freeze takes effect (empirically: MOG2's fast initial
        adaptation absorbed a stationary vehicle within ~10 frames even
        with learningRate=0 applied at frame 10). Real deployments would
        use a captured empty-scene reference (common for fixed traffic
        cameras) or the earliest frames of a longer pre-roll, not
        available in every clip -- flagged as a real limitation when
        unavailable (see docs).
        """
        for _ in range(n_seed_iters):
            self.sub.apply(clean_frame, learningRate=-1)
        self.learning_rate = 0  # freeze immediately -- no further learning

    def detect_blobs(self, frame, edge_margin_px=2):
        """Returns list of {bottom_center: (u,v), area, bbox} for this frame,
        sorted by v ascending (farthest / smallest-v first).

        Blobs whose bounding box is clipped against the bottom frame edge
        are excluded: their true ground-contact point is off-screen (the
        vehicle is too close to the camera / partially out of frame), so
        the detected "bottom" is an artifact of the frame boundary, not a
        real measurement. Verified empirically -- an early, very-close
        vehicle produced a fixed (u, frame_height-1) detection for several
        consecutive frames, contaminating both the reconstructed
        trajectory and the evidence-video trajectory overlay with a
        spurious point. See docs/week3_4_causality_assessment.md.
        """
        frame_h = frame.shape[0]
        fgmask = self.sub.apply(frame, learningRate=self.learning_rate)
        fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]  # drop shadow gray (127)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if y + h >= frame_h - edge_margin_px:
                continue  # clipped against bottom edge -- true contact point is off-screen
            blobs.append({"bottom_center": (x + w / 2.0, y + h - 1), "area": area, "bbox": (x, y, w, h)})
        blobs.sort(key=lambda b: b["bottom_center"][1])
        return blobs


def track_scene(frames, n_vehicles=1, warmup_frames=15, min_area=25,
                 freeze_after_warmup=True, clean_background_frame=None):
    """
    Runs background subtraction across the full frame sequence and
    returns per-vehicle-slot trajectories (list of dicts, one per
    detected vehicle "slot" ordered far-to-near).

    clean_background_frame: if provided (a frame known to contain no
    vehicles), the model is seeded from it and frozen immediately --
    required whenever a vehicle may already be in-frame at t=0 (see
    seed_background docstring). If not provided, falls back to online
    warmup (warmup_frames) + optional freeze, which is only reliable
    when the scene is genuinely empty for the first warmup_frames
    frames.
    """
    tracker = BackgroundSubtractionTracker(min_area=min_area)
    slots = [{"frame_idx": [], "uv": [], "bbox": []} for _ in range(n_vehicles)]

    if clean_background_frame is not None:
        tracker.seed_background(clean_background_frame)
        start_frame = 0
    else:
        start_frame = warmup_frames

    for i, frame in enumerate(frames):
        if clean_background_frame is None and freeze_after_warmup and i == warmup_frames:
            tracker.learning_rate = 0
        blobs = tracker.detect_blobs(frame)
        if i < start_frame:
            continue
        if len(blobs) < n_vehicles:
            continue
        # take the n_vehicles largest-area blobs, then re-sort by v (far->near)
        top = sorted(blobs, key=lambda b: -b["area"])[:n_vehicles]
        top.sort(key=lambda b: b["bottom_center"][1])
        for slot_idx, blob in enumerate(top):
            slots[slot_idx]["frame_idx"].append(i)
            slots[slot_idx]["uv"].append(blob["bottom_center"])
            slots[slot_idx]["bbox"].append(blob["bbox"])

    return slots
