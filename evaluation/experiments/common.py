"""Shared helpers for evaluation experiments (paths, IO, small stats)."""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CACHE_DIR = os.environ.get("STRADIVA_SIM_CACHE", os.path.join(ROOT, "evaluation", "_sim_cache"))
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def read_frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return fps, frames


def save_json(obj, name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    return path


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def wilson_interval(k, n, z=1.645):
    """90% Wilson score interval for a binomial proportion (default z=1.645)."""
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))
