"""Shared survival / PDF model definitions for all kinetics fitters.

Three pipelines fit the same two parametric families through this module:

  single+c:  S(t) = alpha * exp(-t/tau) + c                with alpha + c = 1
  bi+c:      S(t) = alpha_fast * exp(-t/tau_fast)
                  + alpha_slow * exp(-t/tau_slow)
                  + c                                      with alpha_fast + alpha_slow + c = 1

The functions here use the canonical parameter ordering
(alpha[, tau], c) for single and (alpha_fast, tau_fast, alpha_slow,
tau_slow, c) for bi-exp. Each fitter wraps these at its boundary:

  fitting.py             — Method 1 SP-LS and Method 2 KM-LS both call
                           curve_fit with f1_constrained / f2_constrained
                           here, parametrised via (u, c, tau1, tau2) so
                           the optimiser stays in the [0, 1] simplex.
  dwell_time_kinetics.py — Method 2 MLE uses NLLs that call these
                           survival/PDF functions directly.

Method 2 KM-LS reuses Method 1's `fit_single_exp` / `fit_bi_exp` from
`fitting.py` against the grouped Kaplan-Meier survival curve from
`kinetics.events.empirical_survival_grouped`. Method 1 and Method 2 KM-LS
therefore evaluate the *exact same* curve form with the same fitter; only
the empirical inputs differ (origin-aggregated SP vs grouped KM).

Keeping the math in one place means a change to the model family
(e.g. adding a third component) flows to all three fitters automatically.
"""
from __future__ import annotations

import numpy as np


__all__ = [
    "survival_single_c",
    "survival_biexp_c",
    "pdf_single_c",
    "pdf_biexp_c",
    "mean_dwell_biexp_c",
    "half_life",
    "apparent_residence_time",
    "fitted_residence_time",
    "single_c_uvc_to_canonical",
    "biexp_c_uvc_to_canonical",
]


# ── Survival functions S(t) ──────────────────────────────────────────────
def survival_single_c(t, alpha, tau, c):
    """S(t) = alpha * exp(-t / tau) + c."""
    return alpha * np.exp(-np.asarray(t) / tau) + c


def survival_biexp_c(t, alpha_fast, tau_fast, alpha_slow, tau_slow, c):
    """S(t) = alpha_fast e^(-t/tau_fast) + alpha_slow e^(-t/tau_slow) + c."""
    t = np.asarray(t)
    return (alpha_fast * np.exp(-t / tau_fast)
            + alpha_slow * np.exp(-t / tau_slow)
            + c)


# ── Probability density functions f(t) (constant component contributes 0) ─
def pdf_single_c(t, alpha, tau):
    """f(t) = (alpha / tau) * exp(-t / tau).  c contributes no finite density."""
    return (alpha / tau) * np.exp(-np.asarray(t) / tau)


def pdf_biexp_c(t, alpha_fast, tau_fast, alpha_slow, tau_slow):
    """Bi-exp density (constant component contributes no finite density)."""
    t = np.asarray(t)
    return ((alpha_fast / tau_fast) * np.exp(-t / tau_fast)
            + (alpha_slow / tau_slow) * np.exp(-t / tau_slow))


# ── Derived scalars ──────────────────────────────────────────────────────
def mean_dwell_biexp_c(alpha_fast, tau_fast, alpha_slow, tau_slow):
    """Finite-component mean dwell — (alpha_f τ_f + alpha_s τ_s) / (alpha_f+alpha_s)."""
    finite = alpha_fast + alpha_slow
    if finite <= 0:
        return float("nan")
    return (alpha_fast * tau_fast + alpha_slow * tau_slow) / finite


def half_life(tau):
    """Half-life of an exponential decay with time constant `tau`."""
    return tau * np.log(2.0)


def apparent_residence_time(t, S, c=0.0):
    """Integral of (S(t) - c) from t_min to t_max — model-free, finite window."""
    t = np.asarray(t, dtype=float)
    S = np.asarray(S, dtype=float)
    if t.size < 2:
        return float("nan")
    return float(np.trapezoid(S - c, t))


def fitted_residence_time(alpha_fast, tau_fast, alpha_slow, tau_slow):
    """Integral of finite component to infinity: α_f τ_f + α_s τ_s.

    Population-weighted mean of the *mobile* sub-population, with the
    immobile fraction c contributing zero (since α_f + α_s = 1 − c).
    For c > 0, this UNDER-estimates the typical lifetime of a molecule
    that actually leaves — use `mobile_residence_time` for that.
    """
    return alpha_fast * tau_fast + alpha_slow * tau_slow


def mobile_residence_time(alpha_fast, tau_fast, alpha_slow, tau_slow):
    """Conditional mean residence time *given the molecule eventually leaves*:
        ⟨t⟩|mobile = (α_f τ_f + α_s τ_s) / (α_f + α_s)
                   = fitted_residence_time / (1 − c)
    where α_f + α_s = 1 − c (the mobile fraction).

    Excludes the immobilized fraction c. Equals `fitted_residence_time` when
    c = 0; larger by 1/(1−c) when c > 0. This is the form chemists typically
    quote as "the residence time of the bound population".
    """
    denom = alpha_fast + alpha_slow
    if denom <= 0:
        return float("nan")
    return (alpha_fast * tau_fast + alpha_slow * tau_slow) / denom


# ── (u, c) → canonical conversion (Method 1 fitter parameterisation) ─────
def single_c_uvc_to_canonical(tau, c):
    """Convert M1 (tau, c) → canonical (alpha, tau, c) with alpha = 1 - c."""
    return (1.0 - c, tau, c)


def biexp_c_uvc_to_canonical(u, c, tau1, tau2):
    """Convert M1 (u, c, tau1, tau2) → canonical (alpha_fast, tau_fast,
    alpha_slow, tau_slow, c) with alpha_fast + alpha_slow + c = 1.

    The u parameter is the intra-finite split: alpha_fast = u (1-c).
    Caller is responsible for canonicalising tau_fast < tau_slow.
    """
    alpha_fast = u * (1.0 - c)
    alpha_slow = (1.0 - u) * (1.0 - c)
    return (alpha_fast, tau1, alpha_slow, tau2, c)
