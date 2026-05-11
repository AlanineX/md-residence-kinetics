"""Constrained exponential fitting for survival probability data.

Models (all amplitudes non-negative, sum to 1 by construction):
  Single-exp: P(t) = (1-c) * exp(-t/tau) + c
  Bi-exp:     P(t) = alpha1 * exp(-t/tau1) + alpha2 * exp(-t/tau2) + c
              where alpha1 = u*(1-c), alpha2 = (1-u)*(1-c)

Survival forms come from `kinetics.models` so Method 1 and Method 2
evaluate the same mathematical family.
"""

import os
import sys

import numpy as np
from scipy import optimize
from scipy.optimize import brentq

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Import the shared model definitions directly from the file (avoids
# pulling in kinetics/__init__.py which imports MDAnalysis).
import importlib.util as _ilu
_models_spec = _ilu.spec_from_file_location(
    "kinetics_models", os.path.join(_HERE, "kinetics", "models.py"))
_models = _ilu.module_from_spec(_models_spec)
_models_spec.loader.exec_module(_models)
survival_single_c = _models.survival_single_c
survival_biexp_c = _models.survival_biexp_c
apparent_residence_time = _models.apparent_residence_time
fitted_residence_time = _models.fitted_residence_time
mobile_residence_time = _models.mobile_residence_time
single_c_uvc_to_canonical = _models.single_c_uvc_to_canonical
biexp_c_uvc_to_canonical = _models.biexp_c_uvc_to_canonical


# ── Constrained model functions (M1 fitter parameterisation) ────────────
def f1_constrained(x, tau, c):
    """Single-exp with alpha + c = 1; thin wrapper over `survival_single_c`."""
    alpha, tau_can, c_can = single_c_uvc_to_canonical(tau, c)
    return survival_single_c(x, alpha, tau_can, c_can)


def f2_constrained(x, u, c, tau1, tau2):
    """Bi-exp with alpha1 + alpha2 + c = 1; thin wrapper over `survival_biexp_c`."""
    a_fast, tau_f, a_slow, tau_s, c_can = biexp_c_uvc_to_canonical(u, c, tau1, tau2)
    return survival_biexp_c(x, a_fast, tau_f, a_slow, tau_s, c_can)


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
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
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
        # Single-exp mobile residence time = τ (only one finite component,
        # so the conditional mean ⟨t⟩|mobile reduces to τ trivially).
        "mobile_res_time": float(tau_fit),
    }


def fit_bi_exp(t, y, tau_floor=None, tau_ceil=None, reject_degenerate=False):
    """Fit P(t) = alpha1*exp(-t/tau1) + alpha2*exp(-t/tau2) + c  (alpha1+alpha2+c=1).

    Parameterised as (u, c, tau1, tau2) where alpha1=u*(1-c), alpha2=(1-u)*(1-c).
    The constant term `c` represents an unresolved plateau over the observed
    window, not a separately fitted third exponential species. Returns dict
    with raw params + derived metrics, or None.

    Parameters
    ----------
    tau_floor : float, optional
        Minimum allowed value for τ₁ and τ₂. Default 1e-12 (legacy).
    tau_ceil : float, optional
        Maximum allowed value for τ₁ and τ₂. Default None (= ∞, legacy).
        Pass `tau_max_ns` to prevent the fitter from running τ off into
        absurd values when the SP curve has slow / unresolved tail, which
        would otherwise be better captured by `c` (the constant term).
    reject_degenerate : bool, optional
        If True, return None when the fit is degenerate:
          (a) τ₁ or τ₂ ≤ floor   (collapse to δ-spike)
          (b) τ₁ or τ₂ ≥ ceil/2  (runaway tail; should go into c)
          (c) one component amplitude < 1 % (single-exp is more honest)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    n = len(t)
    if n < 4:
        return None

    dt = (t[1] - t[0]) if n > 1 else 0.01
    floor = float(tau_floor) if tau_floor is not None else 1e-12
    ceil_v = float(tau_ceil) if tau_ceil is not None else np.inf
    c0 = max(0.0, min(float(y[-1]), 1.0))
    tau0 = max(np.trapezoid(y, t), dt)
    def _clip_tau(v):
        hi = ceil_v * 0.98 if np.isfinite(ceil_v) else np.inf
        return min(max(float(v), floor * 10), hi) if np.isfinite(hi) else max(float(v), floor * 10)

    # Multi-start matters for bi-exp curves; the model is symmetric and local
    # minima otherwise swap/collapse components easily.
    p0s = []
    for u0 in (0.25, 0.5, 0.75):
        for c_start in (c0, 0.0, min(max(float(y[-1]), 0.0), 0.8)):
            p0s.append((u0, c_start,
                        _clip_tau(tau0 / 4.0), _clip_tau(tau0 * 2.0)))
            p0s.append((u0, c_start,
                        _clip_tau(dt), _clip_tau(max(tau0 * 4.0, dt * 2.0))))

    best = None
    bounds = ([0.0, 0.0, floor, floor], [1.0, 1.0, ceil_v, ceil_v])
    for p0 in p0s:
        try:
            popt_i, pcov_i = optimize.curve_fit(
                f2_constrained, t, y,
                p0=p0,
                bounds=bounds,
                maxfev=50000,
            )
        except Exception:
            continue
        yhat_i = f2_constrained(t, *popt_i)
        rss_i = float(np.sum((y - yhat_i) ** 2))
        if not np.isfinite(rss_i):
            continue
        if best is None or rss_i < best[0]:
            best = (rss_i, popt_i, pcov_i)

    if best is None:
        return None

    _, popt, pcov = best
    u_val, c, tau1, tau2 = popt
    swapped = tau1 > tau2
    if swapped:
        u_val = 1.0 - u_val
        tau1, tau2 = tau2, tau1
        # Transform covariance for params [u, c, tau1, tau2] -> [1-u, c, tau2, tau1].
        J = np.array([[-1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 1.0, 0.0]])
        pcov = J @ pcov @ J.T
    popt = np.array([u_val, c, tau1, tau2], dtype=float)
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
        "mobile_res_time": mobile_residence_time(alpha1, tau1, alpha2, tau2),
        "t_half_fast": t_half_fast,
        "t_half_slow": t_half_slow,
        "t_half_overall": t_half_all,
    }
