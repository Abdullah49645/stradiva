"""
Counterfactual avoidability over an explicit assumption envelope.

What the original module (backend/causality/avoidability.py) did, and why
it was replaced as the source of the verdict:
  * It asked "could the follower have STOPPED before the eventual collision
    POINT" from the earliest observed frame. If the lead vehicle is still
    moving, that is the wrong question (the follower only has to avoid the
    lead, which keeps moving away), and "earliest observed frame" makes the
    answer depend on when somebody happened to start recording.
  * It used ONE (deceleration, reaction time) pair -- 7.0 m/s^2, 1.2 s --
    and reported AVOIDABLE / NOT_AVOIDABLE without asking whether a
    different, equally defensible pair flips the answer.
  * Speed uncertainty never reached the verdict.

This module answers a narrower, defensible question:

  "From the moment the hazard became observable (the lead vehicle's brake
   onset, measured from the video), had the following driver reacted after
   t_r seconds and braked at a_max, would the follower have stayed clear of
   the lead vehicle's actual (observed, then extrapolated) motion?"

and evaluates it on a grid of (t_r, a_max) values AND on Monte Carlo samples
of everything the video measures with uncertainty (geometry, marking prior,
vehicle length, kinematics). Verdict:

  AVOIDABLE            -- avoided in >= 95 % of samples in EVERY envelope cell
  NOT_AVOIDABLE        -- avoided in <= 5 % of samples in EVERY envelope cell
  ASSUMPTION_DEPENDENT -- anything else; reported with the critical
                          reaction time t_r*(a) and its 90 % interval, i.e.
                          WHICH unverifiable assumption decides the answer
  INSUFFICIENT_EVIDENCE -- upstream gates failed (decided by the caller)

Nothing here assigns fault. "Avoidable under assumption X" is a statement
about physics given X, not about a driver.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

DEFAULT_REACTION_GRID_S = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
DEFAULT_DECEL_GRID_MPS2 = [4.0, 5.0, 6.0, 7.0, 8.0]
# The "envelope" over which a verdict must hold. Literature values: perception-
# reaction to an expected/unexpected lead-vehicle brake event ~0.75-2.0 s
# (Green 2000, Transportation Human Factors); full braking on dry asphalt
# ~6.5-8 m/s^2, wet ~4-5.5 m/s^2. See docs/assumptions.md.
ENVELOPE_REACTION_S = (0.75, 2.0)
ENVELOPE_DECEL_MPS2 = (5.0, 8.0)


@dataclass
class TrajectorySample:
    """One Monte Carlo realisation of both vehicles' along-road motion on a
    common time base (seconds), positions of each vehicle's REAR ground contact
    (metres), plus the follower's length."""
    t: np.ndarray
    x_follow: np.ndarray
    v_follow: np.ndarray
    x_lead: np.ndarray
    v_lead: np.ndarray
    a_lead_end: float          # lead acceleration at the end of observation (for extrapolation)
    follow_length_m: float
    hazard_onset_s: float
    t_impact_s: Optional[float]


@dataclass
class CounterfactualResult:
    verdict: str
    narrative: str
    reaction_grid_s: List[float]
    decel_grid_mps2: List[float]
    p_avoid: List[List[float]]                 # [decel][reaction]
    critical_reaction_s: dict = field(default_factory=dict)   # decel -> {median, ci90}
    envelope: dict = field(default_factory=dict)
    hazard_onset: dict = field(default_factory=dict)
    n_samples: int = 0
    assumptions: dict = field(default_factory=dict)


def _lead_position(sample: TrajectorySample, tq):
    """Lead rear-contact position at query times: observed where available,
    then extrapolated with its final observed acceleration (if braking) until
    it stops, else at constant speed."""
    t, x, v = sample.t, sample.x_lead, sample.v_lead
    out = np.interp(tq, t, x)
    after = tq > t[-1]
    if np.any(after):
        dt = tq[after] - t[-1]
        v0 = max(0.0, v[-1])
        a = min(0.0, sample.a_lead_end)
        if a < -1e-6:
            t_stop = v0 / -a
            dts = np.minimum(dt, t_stop)
            out[after] = x[-1] + v0 * dts + 0.5 * a * dts ** 2
        else:
            out[after] = x[-1] + v0 * dt
    return out


def min_gap_if_reacting(sample: TrajectorySample, reaction_s, decel, dt=0.01):
    """Minimum bumper gap (m) between the counterfactual follower and the
    lead. Negative => collision in the counterfactual."""
    t_react = sample.hazard_onset_s + reaction_s
    t0 = sample.t[0]
    # follower state at the reaction instant: observed motion up to then
    if sample.t_impact_s is not None and t_react >= sample.t_impact_s:
        return -1.0        # the reaction would come after the actual impact
    if t_react <= t0:
        xr, vr = sample.x_follow[0] - sample.v_follow[0] * (t0 - t_react), sample.v_follow[0]
        t_before = np.array([t_react])
        gap_before = np.array([np.inf])
    else:
        xr = float(np.interp(t_react, sample.t, sample.x_follow))
        vr = float(np.interp(t_react, sample.t, sample.v_follow))
        t_before = np.arange(max(sample.hazard_onset_s, t0), t_react, dt)
        if len(t_before):
            xf_b = np.interp(t_before, sample.t, sample.x_follow)
            gap_before = _lead_position(sample, t_before) - (xf_b + sample.follow_length_m)
        else:
            gap_before = np.array([np.inf])
    vr = max(0.0, vr)
    t_stop = vr / decel
    tb = np.arange(0.0, t_stop + dt, dt)
    xf = xr + vr * tb - 0.5 * decel * tb ** 2
    xf[tb > t_stop] = xr + vr * t_stop - 0.5 * decel * t_stop ** 2
    gap_after = _lead_position(sample, t_react + tb) - (xf + sample.follow_length_m)
    return float(min(np.min(gap_before), np.min(gap_after)))


def critical_reaction_time(sample: TrajectorySample, decel, lo=0.0, hi=4.0, tol=0.01):
    """Largest reaction time that still avoids the collision at this
    deceleration (bisection on the monotone avoided/not-avoided boundary).
    Returns 0.0 if not avoidable even with an instant reaction."""
    if min_gap_if_reacting(sample, lo, decel) < 0:
        return 0.0
    if min_gap_if_reacting(sample, hi, decel) >= 0:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if min_gap_if_reacting(sample, mid, decel) >= 0:
            lo = mid
        else:
            hi = mid
    return lo


def assess(samples: List[TrajectorySample], reaction_grid=None, decel_grid=None,
           envelope_reaction=ENVELOPE_REACTION_S, envelope_decel=ENVELOPE_DECEL_MPS2,
           hazard_info=None) -> CounterfactualResult:
    reaction_grid = list(reaction_grid or DEFAULT_REACTION_GRID_S)
    decel_grid = list(decel_grid or DEFAULT_DECEL_GRID_MPS2)
    n = len(samples)
    avoided = np.zeros((len(decel_grid), len(reaction_grid), n), bool)
    for k, s in enumerate(samples):
        for i, a in enumerate(decel_grid):
            for j, r in enumerate(reaction_grid):
                avoided[i, j, k] = min_gap_if_reacting(s, r, a) >= 0
    p = avoided.mean(axis=2)
    crit = {}
    for a in decel_grid:
        vals = np.array([critical_reaction_time(s, a) for s in samples])
        crit[f"{a:g}"] = {"median_s": float(np.median(vals)),
                          "ci90_s": [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]}
    in_env = np.array([[envelope_decel[0] <= a <= envelope_decel[1] and envelope_reaction[0] <= r <= envelope_reaction[1]
                        for r in reaction_grid] for a in decel_grid])
    p_env = p[in_env]
    if np.all(p_env >= 0.95):
        verdict = "AVOIDABLE"
    elif np.all(p_env <= 0.05):
        verdict = "NOT_AVOIDABLE"
    else:
        verdict = "ASSUMPTION_DEPENDENT"

    a_ref = 7.0 if 7.0 in decel_grid else decel_grid[len(decel_grid) // 2]
    cr = crit[f"{a_ref:g}"]
    onset_txt = (hazard_info or {}).get("description", "the hazard onset")
    if verdict == "AVOIDABLE":
        narrative = (f"Under every combination of reaction time {envelope_reaction[0]}-{envelope_reaction[1]} s and braking "
                     f"{envelope_decel[0]}-{envelope_decel[1]} m/s^2, braking after {onset_txt} would have kept the "
                     f"following vehicle clear of the lead vehicle in at least 95 % of the reconstructions consistent "
                     f"with the video. At {a_ref:g} m/s^2 a reaction within {cr['median_s']:.2f} s "
                     f"(90 % interval {cr['ci90_s'][0]:.2f}-{cr['ci90_s'][1]:.2f} s) suffices.")
    elif verdict == "NOT_AVOIDABLE":
        narrative = (f"Under every combination of reaction time {envelope_reaction[0]}-{envelope_reaction[1]} s and braking "
                     f"{envelope_decel[0]}-{envelope_decel[1]} m/s^2, braking after {onset_txt} would NOT have kept the "
                     f"following vehicle clear (at most 5 % of consistent reconstructions avoid contact). "
                     f"Even at {a_ref:g} m/s^2 the reaction would have had to come within {cr['median_s']:.2f} s "
                     f"(90 % interval {cr['ci90_s'][0]:.2f}-{cr['ci90_s'][1]:.2f} s).")
    else:
        narrative = (f"The answer depends on assumptions the video cannot verify. At {a_ref:g} m/s^2 braking, the "
                     f"collision is avoided only if the following driver reacted within {cr['median_s']:.2f} s "
                     f"(90 % interval {cr['ci90_s'][0]:.2f}-{cr['ci90_s'][1]:.2f} s) of {onset_txt}; plausible "
                     f"reaction times span {envelope_reaction[0]}-{envelope_reaction[1]} s, so Odolite does not state a "
                     f"single verdict.")
    return CounterfactualResult(
        verdict=verdict, narrative=narrative, reaction_grid_s=reaction_grid, decel_grid_mps2=decel_grid,
        p_avoid=p.round(3).tolist(), critical_reaction_s=crit,
        envelope={"reaction_s": list(envelope_reaction), "decel_mps2": list(envelope_decel)},
        hazard_onset=hazard_info or {}, n_samples=n,
        assumptions={"counterfactual": "follower follows its observed motion until hazard onset + reaction time, "
                                       "then brakes at constant a_max to rest; lead follows its observed motion, "
                                       "extrapolated after the last observation with its final deceleration (or "
                                       "constant speed if not braking)",
                     "gap": "lead rear contact minus (follower rear contact + follower length prior)"})


ONSET_MODEL_FLOOR_S = 0.1


def hinge_onset(t, x, sigma_x=None, t_lo=None, t_hi=None, step=0.01, t_data_max=None):
    """Brake onset by least-squares hinge regression on RAW positions:
    x(t) = x0 + v0 (t - t0) + 0.5 a (t - t0)^2 [t > t0]. Scans t0, returns
    the best t0 and an approximate 90 % interval from the profile chi^2
    (delta = 2.71), widened by a model-error floor. Much less lagged than thresholding a smoothed
    acceleration."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    w = 1.0 / np.asarray(sigma_x, float) ** 2 if sigma_x is not None else np.ones_like(t)
    if t_data_max is not None:          # the model has no standstill: fit only until the vehicle (nearly) stops
        keep = t <= t_data_max
        t, x, w = t[keep], x[keep], w[keep]
    if len(t) < 8:
        return None
    t_lo = t[2] if t_lo is None else max(t_lo, t[2])
    t_hi = t[-3] if t_hi is None else min(t_hi, t[-3])
    cands = np.arange(t_lo, t_hi, step)
    if len(cands) < 2:
        return None
    sse = []
    a_hat = []
    for t0 in cands:
        h = np.clip(t - t0, 0, None)
        A = np.column_stack([np.ones_like(t), t - t0, 0.5 * h ** 2])
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], x * sw, rcond=None)
        r = x - A @ coef
        sse.append(float(np.sum(w * r ** 2)))
        a_hat.append(float(coef[2]))
    sse = np.array(sse)
    k = int(np.argmin(sse))
    dof = max(1, len(t) - 4)
    s2 = sse[k] / dof  # rescale chi^2 by the residual variance (robust to a mis-set sigma_x)
    # 90 % profile interval (delta chi^2 = 2.71 for one parameter), widened by ONSET_MODEL_FLOOR_S on each side:
    # the hinge model (instantaneous constant deceleration) is itself an approximation of how brakes build up,
    # and the samples feeding the verdict must not be more certain about the onset than the model is
    within = cands[(sse - sse[k]) / max(s2, 1e-12) <= 2.71]
    # null model: one constant acceleration throughout. A gradual geometric/contact bias at long
    # range looks like gentle deceleration and is fitted just as well by the null -- then the onset
    # is NOT identified, whatever the hinge says.
    A0 = np.column_stack([np.ones_like(t), t - t.mean(), 0.5 * (t - t.mean()) ** 2])
    sw = np.sqrt(w)
    c0, *_ = np.linalg.lstsq(A0 * sw[:, None], x * sw, rcond=None)
    sse0 = float(np.sum(w * (x - A0 @ c0) ** 2))
    lr = (sse0 - sse[k]) / max(s2, 1e-12)
    return {"t0_s": float(cands[k]), "t0_lo_s": float(within.min()) - ONSET_MODEL_FLOOR_S,
            "t0_hi_s": float(within.max()) + ONSET_MODEL_FLOOR_S,
            "decel_after_mps2": float(-a_hat[k]), "lr_vs_constant_accel": float(lr),
            # a fitted deceleration beyond tyre-road limits (~1.2 g) is a measurement artefact, not a brake onset
            "identified": bool(lr > 13.8 and 1.5 < -a_hat[k] <= 12.0)}
