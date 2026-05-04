"""Constrained exponential fitting for survival probability data.

Models (all amplitudes non-negative, sum to 1 by construction):
  Single-exp: P(t) = (1-c) * exp(-t/tau) + c
  Bi-exp:     P(t) = alpha1 * exp(-t/tau1) + alpha2 * exp(-t/tau2) + c
              where alpha1 = u*(1-c), alpha2 = (1-u)*(1-c)
"""

import numpy as np
from scipy import optimize
from scipy.optimize import brentq


# ── Constrained model functions ──────────────────────────────────────────────

def f1_constrained(x, tau, c):
    """Single-exp with alpha + c = 1."""
    return (1.0 - c) * np.exp(-x / tau) + c


def f2_constrained(x, u, c, tau1, tau2):
    """Bi-exp with alpha1 + alpha2 + c = 1, parameterised via u in [0,1]."""
    alpha1 = u * (1.0 - c)
    alpha2 = (1.0 - u) * (1.0 - c)
    return alpha1 * np.exp(-x / tau1) + alpha2 * np.exp(-x / tau2) + c


# ── Information criteria ─────────────────────────────────────────────────────

def aic(rss, n, k):
    rss = max(rss, np.finfo(float).tiny)
    return n * np.log(rss / n) + 2 * k


def aicc(rss, n, k):
    a = aic(rss, n, k)
    return np.nan if n - k - 1 <= 0 else a + (2 * k * (k + 1)) / (n - k - 1)


def bic(rss, n, k):
    rss = max(rss, np.finfo(float).tiny)
    return n * np.log(rss / n) + k * np.log(n)


# ── Metric utilities ─────────────────────────────────────────────────────────

def r_squared(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def apparent_residence_time(t, P, c=0.0):
    """Integral of (P(t) - c) from 0 to tau_max.  Model-free, finite window."""
    if len(t) < 2:
        return None
    return float(np.trapezoid(np.asarray(P) - c, t))


def fitted_residence_time(alpha1, tau1, alpha2, tau2):
    """Integral of decaying part to infinity: alpha1*tau1 + alpha2*tau2."""
    return alpha1 * tau1 + alpha2 * tau2


def model_free_metrics(t, S, horizons=(1.0, 2.0, 5.0)):
    """Compute tau_max-independent metrics from raw SP curve.

    Returns dict with RMST(t*) and S(t*) at each horizon.
    RMST(t*) = integral_0^{t*} S(tau) dtau  (no model, no c subtraction).
    S(t*)    = raw survival probability at t*.
    """
    t = np.asarray(t, dtype=float)
    S = np.asarray(S, dtype=float)
    result = {}
    for h in horizons:
        mask = t <= h + 1e-6
        if mask.sum() >= 2:
            result[f"RMST_{h:.0f}ns"] = float(np.trapezoid(S[mask], t[mask]))
        else:
            result[f"RMST_{h:.0f}ns"] = None
        idx = np.argmin(np.abs(t - h))
        if abs(t[idx] - h) < 0.02:
            result[f"S_{h:.0f}ns"] = float(S[idx])
        else:
            result[f"S_{h:.0f}ns"] = None
    return result


def half_lives(tau1, tau2):
    ln2 = np.log(2.0)
    return tau1 * ln2, tau2 * ln2


def overall_half_life(u, c, tau1, tau2):
    """Time when P(t) = midpoint between P(0)=1 and plateau c, i.e. (1+c)/2."""
    target = (1.0 + c) / 2.0

    def f(t):
        a1 = u * (1.0 - c)
        a2 = (1.0 - u) * (1.0 - c)
        return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c - target

    t_max = max(tau1, tau2) * 5
    t_test = np.linspace(0, t_max, 1000)
    f_test = np.array([f(t) for t in t_test])
    for i in range(len(f_test) - 1):
        if f_test[i] * f_test[i + 1] < 0:
            try:
                return brentq(f, t_test[i], t_test[i + 1])
            except Exception:
                continue
    return None


# ── Fitting wrappers ─────────────────────────────────────────────────────────

def fit_single_exp(t, y):
    """Fit P(t) = (1-c)*exp(-t/tau) + c.

    Returns dict with raw params [tau, c], all derived metrics, or None.
    """
    n = len(t)
    if n < 2:
        return None

    dt = (t[1] - t[0]) if n > 1 else 0.01
    c0 = max(0.0, min(float(y[-1]), 1.0))
    tau0 = max(np.trapezoid(y, t), dt)

    try:
        popt, pcov = optimize.curve_fit(
            f1_constrained, t, y,
            p0=(max(tau0, 1e-6), c0),
            bounds=([1e-12, 0.0], [np.inf, 1.0]),
            maxfev=30000,
        )
    except Exception:
        return None

    tau_fit, c = popt
    alpha = 1.0 - c
    yhat = f1_constrained(t, *popt)
    rss_val = float(np.sum((y - yhat) ** 2))
    perr = np.sqrt(np.diag(pcov))
    k = 2

    return {
        "params": popt,
        "cov": pcov,
        "rss": rss_val,
        "aic": aic(rss_val, n, k),
        "aicc": aicc(rss_val, n, k),
        "bic": bic(rss_val, n, k),
        "alpha": alpha,
        "tau": tau_fit,
        "c": c,
        "perr_tau": perr[0],
        "perr_c": perr[1],
        "r_squared": r_squared(y, yhat),
        "apparent_res_time": apparent_residence_time(t, y, c=c),
    }


def fit_bi_exp(t, y, tau_floor=None, tau_ceil=None, reject_degenerate=False):
    """Fit P(t) = alpha1*exp(-t/tau1) + alpha2*exp(-t/tau2) + c  (alpha1+alpha2+c=1).

    Parameterised as (u, c, tau1, tau2) where alpha1=u*(1-c), alpha2=(1-u)*(1-c).
    The constant term `c` represents the immobilized fraction (probes still
    bound at t → ∞). Returns dict with raw params + derived metrics, or None.

    Parameters
    ----------
    tau_floor : float, optional
        Minimum allowed value for τ₁ and τ₂. Default 1e-12 (legacy).
    tau_ceil : float, optional
        Maximum allowed value for τ₁ and τ₂. Default None (= ∞, legacy).
        Pass `tau_max_ns` to prevent the fitter from running τ off into
        absurd values when the SP curve has slow / unresolved tail —
        which would otherwise be better captured by `c` (the immobilized
        fraction term).
    reject_degenerate : bool, optional
        If True, return None when the fit is degenerate:
          (a) τ₁ or τ₂ ≤ floor   (collapse to δ-spike)
          (b) τ₁ or τ₂ ≥ ceil/2  (runaway tail; should go into c)
          (c) one component amplitude < 1 % (single-exp is more honest)
    """
    n = len(t)
    if n < 4:
        return None

    dt = (t[1] - t[0]) if n > 1 else 0.01
    floor = float(tau_floor) if tau_floor is not None else 1e-12
    ceil_v = float(tau_ceil) if tau_ceil is not None else np.inf
    c0 = max(0.0, min(float(y[-1]), 1.0))
    tau0 = max(np.trapezoid(y, t), dt)
    # Initial guesses must lie inside [floor, ceil]
    p0_t1 = min(max(tau0 / 3.0, floor * 10), ceil_v * 0.5) if np.isfinite(ceil_v) else max(tau0 / 3.0, floor * 10)
    p0_t2 = min(max(tau0 * 2.0, floor * 10), ceil_v * 0.5) if np.isfinite(ceil_v) else max(tau0 * 2.0, floor * 10)

    try:
        popt, pcov = optimize.curve_fit(
            f2_constrained, t, y,
            p0=(0.5, c0, p0_t1, p0_t2),
            bounds=([0.0, 0.0, floor, floor],
                    [1.0, 1.0, ceil_v, ceil_v]),
            maxfev=50000,
        )
    except Exception:
        return None

    u_val, c, tau1, tau2 = popt
    alpha1 = u_val * (1.0 - c)
    alpha2 = (1.0 - u_val) * (1.0 - c)

    if reject_degenerate:
        # (a) τ collapsed to the floor → fit is bi-exp in name only
        eps_lo = floor * 2.0
        if tau1 < eps_lo or tau2 < eps_lo:
            return None
        # (b) τ ran away to the ceiling → tail belongs in `c`, not τ.
        # Use a 0.9× threshold (was 0.5×) so we only reject fits that
        # parked at the boundary; legitimate slow components (e.g.
        # τ ~ 0.7 × tau_max) remain accepted.
        if np.isfinite(ceil_v):
            eps_hi = ceil_v * 0.9
            if tau1 > eps_hi or tau2 > eps_hi:
                return None
        # (c) one component is effectively zero amplitude
        if alpha1 < 0.01 or alpha2 < 0.01:
            return None
    yhat = f2_constrained(t, *popt)
    rss_val = float(np.sum((y - yhat) ** 2))
    perr = np.sqrt(np.diag(pcov))
    k = 4
    t_half_fast, t_half_slow = half_lives(tau1, tau2)
    t_half_all = overall_half_life(u_val, c, tau1, tau2)

    return {
        "params": popt,
        "cov": pcov,
        "rss": rss_val,
        "aic": aic(rss_val, n, k),
        "aicc": aicc(rss_val, n, k),
        "bic": bic(rss_val, n, k),
        "u": u_val,
        "c": c,
        "tau1": tau1,
        "tau2": tau2,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "perr_u": perr[0],
        "perr_c": perr[1],
        "perr_tau1": perr[2],
        "perr_tau2": perr[3],
        "r_squared": r_squared(y, yhat),
        "apparent_res_time": apparent_residence_time(t, y, c=c),
        "fitted_res_time": fitted_residence_time(alpha1, tau1, alpha2, tau2),
        "t_half_fast": t_half_fast,
        "t_half_slow": t_half_slow,
        "t_half_overall": t_half_all,
    }
