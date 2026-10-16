"""
Unit tests for the CARAI v2 core: each test checks a component against an
analytic answer or a ground truth it did not see. None of them asserts on
an evaluation-suite number (those live in evaluation/results and are
reported, not tested).

Run:  python3 -m pytest -q tests        (from the repo root)
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.causality.counterfactual import (TrajectorySample, assess, critical_reaction_time,  # noqa: E402
                                              hinge_onset, min_gap_if_reacting)
from backend.dashcam.ego_speed import _period_from_centres  # noqa: E402
from backend.reconstruction.kinematics import rts_constant_accel, velocity_kink  # noqa: E402
from backend.verification.skeptic import check_kinematics  # noqa: E402
from cv.geometry.projective import harmonic_horizon_direction  # noqa: E402
from evaluation.sim.world import Camera, CameraSpec  # noqa: E402


# ----------------------------------------------------------------- counterfactual

def _stopped_lead_sample(gap0, v=20.0, L=4.6, hazard=0.0, T=6.0):
    """Follower at constant speed v from x=0; lead stopped with its rear at
    x = L + gap0. Analytic: braking after reaction r at decel a stops the
    follower after v r + v^2 / (2a), so min gap = gap0 - v r - v^2/(2a)."""
    t = np.arange(0.0, T, 0.01)
    return TrajectorySample(t=t, x_follow=v * t, v_follow=np.full_like(t, v),
                            x_lead=np.full_like(t, L + gap0), v_lead=np.zeros_like(t),
                            a_lead_end=0.0, follow_length_m=L, hazard_onset_s=hazard, t_impact_s=None)


@pytest.mark.parametrize("r,a", [(0.75, 7.0), (1.5, 5.0), (2.0, 8.0)])
def test_min_gap_matches_stopping_distance(r, a):
    v, gap0 = 20.0, 70.0
    s = _stopped_lead_sample(gap0, v)
    expected = gap0 - v * r - v * v / (2 * a)
    assert min_gap_if_reacting(s, r, a) == pytest.approx(expected, abs=0.25)


def test_critical_reaction_time_analytic():
    v, gap0, a = 20.0, 60.0, 7.0
    s = _stopped_lead_sample(gap0, v)
    expected = (gap0 - v * v / (2 * a)) / v
    assert critical_reaction_time(s, a) == pytest.approx(expected, abs=0.02)


def test_reaction_after_actual_impact_is_not_avoidance():
    s = _stopped_lead_sample(20.0)
    s.t_impact_s = 1.0
    assert min_gap_if_reacting(s, 1.5, 8.0) < 0


@pytest.mark.parametrize("gap0,verdict", [(120.0, "AVOIDABLE"), (25.0, "NOT_AVOIDABLE"), (50.0, "ASSUMPTION_DEPENDENT")])
def test_envelope_verdicts(gap0, verdict):
    # v = 20 m/s: stopping needs 20 r + 400/(2a): 40 m at (0.75 s, 8) ... 80 m at (2.0 s, 5)
    samples = [_stopped_lead_sample(gap0 + d) for d in np.linspace(-1, 1, 5)]
    assert assess(samples).verdict == verdict


# ----------------------------------------------------------------- onset / impact

def _brake_track(t0=1.5, v0=20.0, a=6.0, noise=0.03, seed=0, T=3.5, fps=30):
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, 1 / fps)
    h = np.clip(t - t0, 0, None)
    x = v0 * t - 0.5 * a * h ** 2
    return t, x + rng.normal(0, noise, len(t)), np.full(len(t), noise)


def test_hinge_onset_recovers_brake_onset():
    t, x, s = _brake_track()
    out = hinge_onset(t, x, s)
    assert out["identified"]
    assert out["t0_s"] == pytest.approx(1.5, abs=0.1)
    assert out["decel_after_mps2"] == pytest.approx(6.0, rel=0.15)


def test_hinge_onset_not_identified_for_constant_acceleration():
    rng = np.random.default_rng(1)
    t = np.arange(0, 3.5, 1 / 30)
    x = 20 * t - 0.5 * 2.0 * t ** 2 + rng.normal(0, 0.03, len(t))
    out = hinge_onset(t, x, np.full(len(t), 0.03))
    assert not out["identified"]


def test_velocity_kink_detects_impact_but_not_smooth_braking():
    rng = np.random.default_rng(2)
    t = np.arange(0, 3.0, 1 / 30)
    tk, dv = 1.8, 6.0
    x_imp = 15 * t - dv * np.clip(t - tk, 0, None) + rng.normal(0, 0.03, len(t))
    k = velocity_kink(t, x_imp, np.full(len(t), 0.03))
    assert k["significant"] and k["t_s"] == pytest.approx(tk, abs=0.07)
    assert k["dv_mps"] == pytest.approx(-dv, rel=0.2)
    t2, x2, s2 = _brake_track(t0=1.5, a=6.0, T=3.0)
    assert not velocity_kink(t2, x2, s2)["significant"]


def test_rts_constant_accel_recovers_acceleration():
    rng = np.random.default_rng(3)
    t = np.arange(0, 3, 1 / 30)
    x = 25 * t - 0.5 * 5.0 * t ** 2
    xs, Ps = rts_constant_accel(t, x + rng.normal(0, 0.05, len(t)), np.full(len(t), 0.05 ** 2), q_jerk=40.0)
    mid = slice(15, -15)
    assert np.median(xs[mid, 1] - (25 - 5.0 * t[mid])) == pytest.approx(0, abs=0.15)
    assert np.median(xs[mid, 2]) == pytest.approx(-5.0, abs=0.6)


# ----------------------------------------------------------------- projective geometry

def _line_angle(vp, p):
    d = np.asarray(p) - vp
    return math.degrees(math.atan2(d[0], d[1]))      # convention of cv.geometry.projective.direction_vector


@pytest.mark.parametrize("roll", [-4.0, 0.0, 2.5, 7.0])
def test_harmonic_horizon_gives_camera_roll(roll):
    """Three equally spaced parallel road lines -> horizon direction -> roll,
    in closed form, against a synthetic camera with known roll."""
    cam = Camera(CameraSpec(image_w=1280, image_h=720, focal_px=1100, height_m=7.0, pitch_deg=9.0,
                            roll_deg=roll, yaw_deg=0.0, x_m=0.0, y_m=-6.0))
    far, near = 5000.0, 25.0
    W = 1.85
    vp = cam.project([[1e7, 0.0, 0.0]])[0][0]
    angs = [_line_angle(vp, cam.project([[near, y, 0.0]])[0][0]) for y in (-W, 0.0, W)]
    # order left/centre/right by image angle as the pipeline does
    a_l, a_c, a_r = sorted(angs, reverse=True)
    hd = harmonic_horizon_direction(vp, a_l, a_c, a_r)
    est_roll = -math.degrees(math.atan2(hd[1], hd[0]))
    # true horizon: two far points along the road plane at different lateral offsets
    h = cam.project([[far, -2000.0, 0.0], [far, 2000.0, 0.0]])[0]
    true_dir = h[1] - h[0]
    true_roll = -math.degrees(math.atan2(true_dir[1], true_dir[0]))
    assert est_roll == pytest.approx(true_roll, abs=0.05)


# ----------------------------------------------------------------- dashcam period

def test_period_from_centres_tolerates_missing_dash():
    rng = np.random.default_rng(4)
    period = 0.012
    k = np.array([0, 1, 2, 4, 5, 6, 7])           # dash 3 is worn away
    cen = 0.05 + period * k + rng.normal(0, 1e-5, len(k))
    p, resid = _period_from_centres(cen)
    assert p == pytest.approx(period, rel=0.01)


# ----------------------------------------------------------------- Skeptic scoping

def _kin(a_peak, v=20.0, n=60, vid=0):
    """Trajectory braking at a_peak for 0.6 s (sustained), constant speed otherwise."""
    t = np.linspace(0, 2, n)
    a = np.where((t >= 0.8) & (t < 1.4), a_peak, 0.0)
    vel = v + np.concatenate([[0], np.cumsum(a[:-1] * np.diff(t))])
    return {"t": t, "v": vel, "a": a, "sv": np.full(n, 0.2), "sa": np.full(n, 0.5), "vehicle_id": vid}


def test_skeptic_kinematics_ignores_one_frame_smoother_overshoot():
    k = _kin(-6.0, vid=1)
    k["a"] = k["a"].copy()
    k["a"][30] = -40.0                     # a spike in the smoothed acceleration, speeds unchanged
    assert check_kinematics([k], subjects={1})["status"] == "PASS"


def test_skeptic_kinematics_blocks_only_on_subject_vehicles():
    ok, bad = _kin(-6.0, vid=1), _kin(-30.0, vid=7)
    r = check_kinematics([ok, _kin(-5.0, vid=2), bad], subjects={1, 2})
    assert r["status"] == "ADVISORY" and r["withheld_vehicles"] == [7]
    r = check_kinematics([ok, _kin(-5.0, vid=2), bad], subjects={1, 7})
    assert r["status"] == "DISPUTE"
    # no interacting pair: one bad trajectory of three is withheld, not a global dispute
    assert check_kinematics([ok, _kin(-5.0, vid=2), bad])["status"] == "ADVISORY"
    assert check_kinematics([_kin(-30.0, vid=1), _kin(-25.0, vid=2), ok])["status"] == "DISPUTE"
