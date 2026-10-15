"""
Road-plane kinematics with explicit measurement noise.

Replaces backend/reconstruction/trajectory.py's Savitzky-Golay smoother,
whose window is 35 % of the WHOLE track: for a constant-speed synthetic car
that makes speed look extremely accurate, but it smears a 0.3 s braking
onset across a second or more and invents acceleration at the track ends.

Here: a constant-acceleration Kalman filter + Rauch-Tung-Striebel smoother
along the road (X) and a constant-velocity one across it (Y).

* Measurement noise is HETEROSCEDASTIC and comes from the geometry: a pixel
  of ground-contact jitter is centimetres near the camera and metres far
  away. sigma_X per frame = |dX/d(u,v)| * sigma_px, from the homography's
  local Jacobian. Far, noisy positions are automatically trusted less.
* Process noise is white jerk (q_jerk, m^2/s^5): the only tuning knob,
  chosen on the development scenes (evaluation/experiments) and fixed
  before the held-out evaluation.
* Output includes per-frame standard deviations of position, speed and
  acceleration -- the measurement part of the uncertainty budget.
"""
from __future__ import annotations

import numpy as np

from backend.geometry.road_geometry import image_to_road


def _jacobian(H, uv, eps=0.5):
    u, v = uv
    p = image_to_road(H, [[u - eps, v], [u + eps, v], [u, v - eps], [u, v + eps]])
    dXdu = (p[1] - p[0]) / (2 * eps)
    dXdv = (p[3] - p[2]) / (2 * eps)
    return np.column_stack([dXdu, dXdv])       # rows: X, Y ; cols: u, v


def rts_constant_accel(t, z, r, q_jerk, gate_sigma=5.0, return_rejected=False):
    """Kalman filter + RTS smoother, state [pos, vel, acc], measurements z
    with variances r at (possibly irregular) times t. Measurements whose
    innovation exceeds gate_sigma standard deviations are rejected (an
    identity swap or a blob fragment must not drag the trajectory)."""
    n = len(t)
    xs_f = np.zeros((n, 3))
    Ps_f = np.zeros((n, 3, 3))
    xs_p = np.zeros((n, 3))
    Ps_p = np.zeros((n, 3, 3))
    Fs = np.zeros((n, 3, 3))
    x = np.array([z[0], (z[min(3, n - 1)] - z[0]) / max(t[min(3, n - 1)] - t[0], 1e-3), 0.0])
    P = np.diag([r[0], 25.0, 25.0])
    rejected = []
    Hm = np.array([[1.0, 0.0, 0.0]])
    for k in range(n):
        dt = t[k] - t[k - 1] if k > 0 else 0.0
        F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]])
        Q = q_jerk * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                               [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                               [dt ** 3 / 6, dt ** 2 / 2, dt]])
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        Fs[k] = F
        xs_p[k], Ps_p[k] = x, P
        S = (Hm @ P @ Hm.T)[0, 0] + r[k]
        innov = z[k] - x[0]
        if k >= 4 and abs(innov) > gate_sigma * np.sqrt(S):
            rejected.append(k)
        else:
            K = (P @ Hm.T)[:, 0] / S
            x = x + K * innov
            P = (np.eye(3) - np.outer(K, Hm)) @ P
        xs_f[k], Ps_f[k] = x, P
    xs, Ps = xs_f.copy(), Ps_f.copy()
    for k in range(n - 2, -1, -1):
        F = Fs[k + 1]
        C = Ps_f[k] @ F.T @ np.linalg.inv(Ps_p[k + 1])
        xs[k] = xs_f[k] + C @ (xs[k + 1] - xs_p[k + 1])
        Ps[k] = Ps_f[k] + C @ (Ps[k + 1] - Ps_p[k + 1]) @ C.T
    if return_rejected:
        return xs, Ps, rejected
    return xs, Ps


def rts_constant_velocity(t, z, r, q_acc):
    n = len(t)
    xs_f, Ps_f, xs_p, Ps_p, Fs = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros((n, 2, 2))
    x = np.array([z[0], 0.0])
    P = np.diag([r[0], 4.0])
    for k in range(n):
        dt = t[k] - t[k - 1] if k > 0 else 0.0
        F = np.array([[1, dt], [0, 1]])
        Q = q_acc * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        Fs[k] = F
        xs_p[k], Ps_p[k] = x, P
        S = P[0, 0] + r[k]
        K = P[:, 0] / S
        x = x + K * (z[k] - x[0])
        P = P - np.outer(K, P[0, :])
        xs_f[k], Ps_f[k] = x, P
    xs, Ps = xs_f.copy(), Ps_f.copy()
    for k in range(n - 2, -1, -1):
        F = Fs[k + 1]
        C = Ps_f[k] @ F.T @ np.linalg.inv(Ps_p[k + 1])
        xs[k] = xs_f[k] + C @ (xs[k + 1] - xs_p[k + 1])
        Ps[k] = Ps_f[k] + C @ (Ps[k + 1] - Ps_p[k + 1]) @ C.T
    return xs, Ps


def road_kinematics(times_s, uv_points, H, sigma_px=0.8, q_jerk=40.0, q_lat=1.0, scale=1.0):
    """times_s: measurement times; uv_points: ground-contact pixels.
    scale multiplies road coordinates (used to apply a sampled marking prior).
    Returns dict of arrays (t, x, v, a, y, vy and their sigmas)."""
    t = np.asarray(times_s, float)
    uv = np.asarray(uv_points, float)
    if len(t) < 4:
        return None
    xy = image_to_road(H, uv) * scale
    rx, ry = [], []
    for p in uv:
        J = _jacobian(H, p) * scale
        rx.append((J[0, 0] ** 2 + J[0, 1] ** 2) * sigma_px ** 2)
        ry.append((J[1, 0] ** 2 + J[1, 1] ** 2) * sigma_px ** 2)
    rx, ry = np.array(rx), np.array(ry)
    xs, Ps, rej = rts_constant_accel(t, xy[:, 0], rx, q_jerk, return_rejected=True)
    ys, Py = rts_constant_velocity(t, xy[:, 1], ry, q_lat)
    return {
        "t": t, "x_raw": xy[:, 0], "y_raw": xy[:, 1], "sigma_x_meas": np.sqrt(rx),
        "x": xs[:, 0], "v": xs[:, 1], "a": xs[:, 2],
        "sx": np.sqrt(Ps[:, 0, 0]), "sv": np.sqrt(Ps[:, 1, 1]), "sa": np.sqrt(Ps[:, 2, 2]),
        "y": ys[:, 0], "vy": ys[:, 1], "sy": np.sqrt(Py[:, 0, 0]),
        "rejected_idx": np.array(rej, int),
    }


def velocity_kink(t, x, sigma_x=None, t_lo=None, t_hi=None, step=0.01, min_dv=1.5, z=3.0):
    """Impact detector: least-squares fit with a sudden SPEED change at tk,
    x = c0 + c1 tau + c2 tau^2 + c3 tau+ + c4 tau+^2 (tau = t - tk). A
    collision shows up as a significant negative c3 (speed lost in one frame)
    in the motion of a vehicle that stays visible. Returns dict or None."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    sel = np.ones(len(t), bool)
    if t_lo is not None:
        sel &= t >= t_lo - 0.6
    if t_hi is not None:
        sel &= t <= t_hi + 0.6
    t, x = t[sel], x[sel]
    w = (1.0 / np.asarray(sigma_x, float)[sel] ** 2) if sigma_x is not None else np.ones_like(t)
    if len(t) < 12:
        return None
    best = None
    lo = max(t[4], t_lo if t_lo is not None else t[4])
    hi = min(t[-5], t_hi if t_hi is not None else t[-5])
    for tk in np.arange(lo, hi, step):
        tau = t - tk
        tp = np.clip(tau, 0, None)
        A = np.column_stack([np.ones_like(t), tau, tau ** 2, tp, tp ** 2])
        sw = np.sqrt(w)
        Aw = A * sw[:, None]
        coef, *_ = np.linalg.lstsq(Aw, x * sw, rcond=None)
        r = x - A @ coef
        sse = float(np.sum(w * r ** 2))
        if best is None or sse < best[0]:
            dof = max(1, len(t) - 5)
            s2 = sse / dof
            cov = s2 * np.linalg.pinv(Aw.T @ Aw)
            best = (sse, tk, coef, float(np.sqrt(max(cov[3, 3], 1e-12))))
    if best is None:
        return None
    sse, tk, coef, sd = best
    dv = float(coef[3])
    return {"t_s": float(tk), "dv_mps": dv, "dv_sigma": sd,
            "significant": bool(dv < -min_dv and abs(dv) > z * sd)}


def braking_onset(kin, a_thresh=2.0, min_duration_s=0.2):
    """First time the smoothed deceleration exceeds a_thresh with ~95 %
    confidence (a + 1.645*sigma_a < -a_thresh) for min_duration_s.
    Returns dict or None. Also returns the uncertainty window: the
    earliest time the point estimate crossed, and the time the
    confident crossing happened."""
    t, a, sa = kin["t"], kin["a"], kin["sa"]
    conf = (a + 1.645 * sa) < -a_thresh
    point = a < -a_thresh
    need = max(1, int(round(min_duration_s / np.median(np.diff(t))))) if len(t) > 1 else 1
    run = 0
    for i, c in enumerate(conf):
        run = run + 1 if c else 0
        if run >= need:
            i0 = i - need + 1
            # earliest point-estimate crossing that leads into this run
            j = i0
            while j > 0 and point[j - 1]:
                j -= 1
            return {"t_confident_s": float(t[i0]), "t_earliest_s": float(t[j]),
                    "peak_decel_mps2": float(-np.min(a[i0:])) if i0 < len(a) else None}
    return None
