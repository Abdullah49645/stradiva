"""
Stage: vehicle tracking (brief section 13).

For Week-1 feasibility we use a deliberately simple, swappable detector
(color-blob segmentation, since the synthetic vehicle is chromatically
distinct from the road/paint) whose only job is to prove out the
*downstream* geometry -> trajectory -> speed math. Section 13 calls for
OpenCV tracking/optical-flow/Kalman filtering on real detections; that
upgrade is explicitly deferred to Week 3 (see docs/week1_feasibility.md)
once a real or more realistic vehicle detector is wired in.

Per section 11, we track the *ground-contact point*, not the bbox
center: for this camera model the rear ground-contact point is the
bottom-center pixel of the vehicle silhouette (see reasoning in
run_feasibility.py).
"""
import cv2
import numpy as np


def track_ground_contact_point(bgr, prev_point=None, search_radius=250,
                                color="red"):
    """
    Detect a colored vehicle blob and return its bottom-center pixel
    (u, v) as the ground-contact estimate, plus a confidence score based
    on blob size. Returns (None, 0.0) if not found.

    color: "red" (default, single-vehicle scenes) or "green" (lead
    vehicle in two-vehicle collision scenes -- see
    evaluation/scripts/generate_collision_scene.py).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if color == "red":
        lower1, upper1 = np.array([0, 100, 60]), np.array([10, 255, 255])
        lower2, upper2 = np.array([170, 100, 60]), np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    elif color == "green":
        lower, upper = np.array([40, 80, 60]), np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        raise ValueError(f"unsupported color: {color}")
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    def score(c):
        area = cv2.contourArea(c)
        if prev_point is None:
            return area
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2.0, y + h
        dist = np.hypot(cx - prev_point[0], cy - prev_point[1])
        return area if dist < search_radius else -1

    best = max(contours, key=score)
    if cv2.contourArea(best) < 15:
        return None, 0.0
    x, y, w, h = cv2.boundingRect(best)
    bottom_center = (x + w / 2.0, y + h - 1)
    confidence = min(1.0, cv2.contourArea(best) / 400.0)
    return bottom_center, confidence
