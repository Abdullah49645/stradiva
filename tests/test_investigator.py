"""Investigator pieces that can be tested without rendering a scene."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.geometry.marking_standards import BY_KEY, STANDARDS, candidates_for_duty  # noqa: E402
from backend.investigator.inputs import validate_inputs  # noqa: E402
from backend.investigator.investigator import _near_event, find_camera_shift  # noqa: E402


def _scene(rng):
    img = np.full((360, 640, 3), 90, np.uint8)
    for k in range(12):
        cv2.line(img, (int(rng.uniform(0, 640)), 360), (320, 120), (230, 230, 230), 3)
    cv2.rectangle(img, (40, 40), (140, 110), (40, 60, 160), -1)
    return img


def test_camera_shift_found_once_and_not_on_still_clip():
    rng = np.random.default_rng(0)
    base = _scene(rng)
    noise = lambda: np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)  # noqa: E731
    M = cv2.getRotationMatrix2D((320, 180), 1.5, 1.0)
    M[:, 2] += (6, 3)
    frames = [noise() for _ in range(45)] + [cv2.warpAffine(noise(), M, (640, 360), borderMode=cv2.BORDER_REPLICATE) for _ in range(45)]
    k, _, why = find_camera_shift(frames, 30.0)
    assert k is not None and abs(k - 45) <= 1, why
    k2, _, why2 = find_camera_shift([noise() for _ in range(90)], 30.0)
    assert k2 is None, why2


def test_standard_candidates_by_duty():
    keys = [s.key for s in candidates_for_duty(0.40, 0.06)]   # what blur does to 6 m + 12 m dashes
    assert keys[0] == "at_rural" and "at_motorway" in keys and "us_mutcd" not in keys
    # the measurement is biased up, so a measured 0.33 cannot rule out the 25 % standards
    assert {"us_mutcd", "carai_synthetic", "at_motorway"} <= {s.key for s in candidates_for_duty(0.33, 0.06)}
    assert all(s.source for s in STANDARDS)


def test_inputs_validation():
    assert validate_inputs({"fps": "25", "marking_standard": "fr_t1"}) == ({"fps": 25.0, "marking_standard": "fr_t1"}, None)
    assert validate_inputs({"fps": 0})[1] and validate_inputs({"marking_standard": "x"})[1] and validate_inputs({"y": 1})[1]
    assert validate_inputs(None) == ({}, None)
    assert set(BY_KEY) >= {"us_mutcd", "caltrans_freeway"}


def test_quarantine_refused_near_the_event():
    t = np.linspace(0, 4, 121)
    kf = {"t": t, "x": 10 * t}
    kl = {"t": t, "x": 20 + 6 * t}           # gap closes to 20 - 4*4 - 4.6 < 0 near the end
    early = {int(round(x * 30)) for x in np.linspace(0.5, 0.9, 13)}
    late = {int(round(x * 30)) for x in np.linspace(3.2, 3.6, 13)}
    assert _near_event(early, 30.0, (kf, kl)) is None
    assert _near_event(late, 30.0, (kf, kl)) is not None
