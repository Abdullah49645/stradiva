"""
COOL/Graviton benchmark target (brief section 21 / Best Use of COOL
award). Architecture-portable -- no ARM-specific code, so this runs
unmodified on an x86 instance and a Graviton (arm64) instance; only the
timing numbers should differ.

WHY THIS LOOP AND NOT SOMETHING ELSE: profiled the full single-vehicle
pipeline (evaluation/scripts/run_feasibility.py) with cProfile first
(see docs/cool_graviton_assessment.md for the full profile). Three
candidate targets came out of that, and this is the only one that's a
defensible COOL case rather than a forced one:

  - cv2.VideoCapture.read() dominates total wall time (2.42s/3.30s) but
    is video decode/demux, not an "OpenCV operation" in the sense the
    award means -- mostly libavcodec software decoding, a different
    optimization story than vectorized image-processing throughput.
  - The one-time geometry/agent search (line detection, homography fit)
    runs once per video regardless of length -- real CPU time, but it
    doesn't scale with data volume, so it's a weak "COOL saves you
    money at scale" story.
  - THIS loop -- cvtColor + inRange + morphologyEx + findContours,
    called once per frame inside vehicle tracking -- is genuinely
    vectorized, genuinely repeated (scales linearly with video length /
    fleet size), and is exactly the "image preprocessing, feature
    operations" category the brief names as a good COOL candidate.

Usage: `python benchmark_tracking_loop.py` on each instance type, then
compare `frames_per_second` and `ms_per_frame`.
"""
import platform
import time

import cv2
import numpy as np

N_FRAMES = 300
FRAME_SIZE = (720, 1280, 3)  # matches the synthetic dataset's resolution


def make_synthetic_frame(rng, w=FRAME_SIZE[1], h=FRAME_SIZE[0]):
    """A frame with a red blob in a plausible location, so the real
    detection/contour code path executes (not just an early-out on an
    empty mask) -- otherwise the benchmark would understate real cost."""
    frame = (rng.integers(60, 180, size=FRAME_SIZE, dtype=np.uint8))
    cx, cy = rng.integers(200, w - 200), rng.integers(200, h - 100)
    cv2.circle(frame, (int(cx), int(cy)), 40, (0, 0, 220), -1)  # BGR red-ish blob
    return frame


def run_hot_loop(frame):
    """Exact operation sequence from cv/tracking/vehicle_tracker.py's
    track_ground_contact_point (red-vehicle path) -- copied here rather
    than imported so this file has zero project dependencies beyond
    opencv-python/numpy and can be dropped onto a bare benchmark
    instance without the rest of the repo."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0, 100, 60]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 100, 60]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def main():
    rng = np.random.default_rng(42)
    frames = [make_synthetic_frame(rng) for _ in range(N_FRAMES)]

    # warm-up (page faults, cache warming, first-call overhead) excluded from the timed run
    for f in frames[:20]:
        run_hot_loop(f)

    t0 = time.perf_counter()
    for f in frames:
        run_hot_loop(f)
    elapsed = time.perf_counter() - t0

    print(f"platform: {platform.machine()} / {platform.processor() or 'unknown'}")
    print(f"opencv version: {cv2.__version__}")
    print(f"frames: {N_FRAMES}  frame_size: {FRAME_SIZE}")
    print(f"total time: {elapsed:.3f}s")
    print(f"ms_per_frame: {1000 * elapsed / N_FRAMES:.3f}")
    print(f"frames_per_second: {N_FRAMES / elapsed:.1f}")


if __name__ == "__main__":
    main()
