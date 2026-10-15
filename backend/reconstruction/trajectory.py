"""
Stage: motion reconstruction (brief section 14).

Takes noisy per-frame ground-plane positions and produces smoothed
position/velocity estimates. Explicitly separates observed (raw tracked
position), derived (smoothed position), and estimated (velocity,
acceleration) quantities per section 14's requirement.
"""
import numpy as np
from scipy.signal import savgol_filter


def smooth_trajectory(t_s, x_m, y_m, window_frac=0.35, polyorder=2):
    """
    Savitzky-Golay smoothing (preserves motion trends better than a
    moving average while suppressing per-frame tracking jitter).
    Falls back to raw values if too few samples for the filter window.
    """
    n = len(x_m)
    if n == 0:
        return np.array([]), np.array([])
    window = int(max(polyorder + 2, round(n * window_frac)))
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window <= polyorder or window < 5:
        return np.array(x_m), np.array(y_m)
    x_s = savgol_filter(x_m, window, polyorder)
    y_s = savgol_filter(y_m, window, polyorder)
    return x_s, y_s


def velocity_from_position(t_s, x_s, y_s):
    """Central-difference velocity (m/s) from smoothed position."""
    if len(x_s) < 2:
        empty = np.array([])
        return empty, empty, empty
    vx = np.gradient(x_s, t_s)
    vy = np.gradient(y_s, t_s)
    speed = np.hypot(vx, vy)
    return vx, vy, speed


def acceleration_from_velocity(t_s, speed):
    return np.gradient(speed, t_s)


def relative_displacement(x_s, y_s):
    """
    Displacement from the first observed sample, along-road and lateral.

    Why this exists: absolute world position depends on correctly knowing
    *where on the road* the homography's world origin sits (see the
    dash-anchoring failure mode in docs/week1_feasibility.md -- a wrong
    anchor assumption produces a constant offset in absolute X that does
    NOT affect displacement, velocity, or speed). Relative displacement is
    therefore reported as the higher-confidence quantity whenever no
    independently-verified anchor (a stationary landmark of known
    position, e.g.) is available; absolute position should be labeled
    low-confidence in that case, not silently trusted.
    """
    return x_s - x_s[0], y_s - y_s[0]
