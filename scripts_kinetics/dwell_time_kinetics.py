#!/usr/bin/env python3
"""Dwell-time kinetics — single-species, event-based, MLE-fitted.

Reads T1 _intermediate/contacts/chain_X.npz CSR-style files, reconstructs
per-ADP-molecule binary occupancy series, extracts dwell intervals, and
fits the dwell distribution with single- and bi-exponential MIXTURE
models using **maximum likelihood estimation** (handles right-censored
events properly).

Outputs (results_v3/dwell_kinetics/), matching the residence-time pipeline
style (adp_0216/, etc.):

  bi_exp_fitting_results.csv      — one row per buffer × chain (+ pooled)
  single_exp_fitting_results.csv  — one row per buffer × chain (+ pooled)
  kon_koff_summary.csv            — explicit k_on, k_off per chain × buffer
  summary_<buf>.txt               — human-readable per-buffer summary
  dwell_events_<buf>.csv          — every event (chain, resid, t, dwell, censored)
  plots/dwell_<buf>.svg           — histogram + PDF + survival fits
  plots/dwell_combined.svg        — both buffers overlay

Models
------
  Single-exp PDF:   f(t) = (1/τ) exp(-t/τ)
  Single-exp survival:  S(t) = exp(-t/τ)
  Bi-exp PDF:       f(t) = (π/τ₁) exp(-t/τ₁) + ((1-π)/τ₂) exp(-t/τ₂)
  Bi-exp survival:  S(t) = π exp(-t/τ₁) + (1-π) exp(-t/τ₂)

Parameter SEs from inverse Hessian of -log L (observed Fisher information).
AIC/BIC computed from -2logL + 2k / k log N.

Usage:
    python dwell_time_kinetics.py
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Sequence

import numpy as np
from scipy import optimize


# ── Constants ──────────────────────────────────────────────────────────────
RES = "/home/alan/working/groel_new/results_v3"
DT_NS = 0.10                # 10× stride on 0.01 ns trajectory
N_FRAMES = 5001             # frames in stored contacts npz
T_TOTAL_NS = (N_FRAMES - 1) * DT_NS  # 500.0 ns
N_ADP = 300                 # bulk ADP count (identical in both buffers)
BOX_NM = 19.1               # cubic edge
# 1 nm^3 = 1e-24 L
V_L = BOX_NM ** 3 * 1e-24
N_AVO = 6.02214076e23
BULK_M = N_ADP / (N_AVO * V_L)  # ≈ 0.0715 M = 71.5 mM
CHAINS = list("ABCDEFG")
BUFFERS = ("EDDA", "AMAC")
INTERMITTENCY = 0           # frames a molecule is allowed to leave & come back
                            # without breaking a dwell event (matches T1)
OUT_DIR = os.path.join(RES, "dwell_kinetics")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
HORIZONS = (1.0, 2.0, 5.0)


# ── Per-molecule occupancy reconstruction ──────────────────────────────────
def load_occupancy(npz_path: str) -> dict[int, np.ndarray]:
    """Return {adp_resid: binary array of length N_FRAMES}."""
    d = np.load(npz_path, allow_pickle=True)
    fi = d["frame_indices"]
    off = d["offsets"]
    ri = d["resindices"]
    if len(ri) == 0:
        return {}

    by_mol: dict[int, list[int]] = {}
    for i, frame in enumerate(fi):
        for resid in ri[off[i]:off[i + 1]]:
            by_mol.setdefault(int(resid), []).append(int(frame))

    occ = {}
    for resid, frames in by_mol.items():
        a = np.zeros(N_FRAMES, dtype=np.uint8)
        local = (np.asarray(frames) // 10).astype(int)
        a[local] = 1
        occ[resid] = a
    return occ


def runs_to_events(arr: np.ndarray, intermittency: int = 0):
    """Return list of (start_idx, end_idx_exclusive). Merges gaps ≤ intermittency."""
    if not arr.any():
        return []
    if intermittency > 0:
        a = arr.copy()
        idx = np.where(np.diff(np.concatenate([[0], a, [0]])) != 0)[0]
        starts, stops = idx[::2], idx[1::2]
        for k in range(len(stops) - 1):
            gap = starts[k + 1] - stops[k]
            if 0 < gap <= intermittency:
                a[stops[k]:starts[k + 1]] = 1
        arr = a
    edges = np.diff(np.concatenate([[0], arr.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    return list(zip(starts.tolist(), stops.tolist()))


def collect_events(buffer: str):
    """Return (events_list, arrivals_per_chain dict)."""
    events = []
    arrivals = {}
    for ch in CHAINS:
        p = os.path.join(RES, f"T1_per_chain_pocket_{buffer}",
                         "_intermediate", "contacts", f"chain_{ch}.npz")
        if not os.path.exists(p):
            arrivals[ch] = 0
            continue
        occ = load_occupancy(p)
        n_arr = 0
        for resid, arr in occ.items():
            for s, e in runs_to_events(arr, INTERMITTENCY):
                left_cens = (s == 0)
                right_cens = (e == N_FRAMES)
                events.append({
                    "chain": ch,
                    "adp_resid": resid,
                    "t_start_ns": s * DT_NS,
                    "dwell_ns": (e - s) * DT_NS,
                    "left_cens": int(left_cens),
                    "right_cens": int(right_cens),
                })
                if not left_cens:
                    n_arr += 1
        arrivals[ch] = n_arr
    return events, arrivals


# ── Likelihoods (single & bi-exp mixture) ──────────────────────────────────
def _safe_log(x):
    return np.log(np.maximum(x, 1e-300))


def nll_single(theta: Sequence[float], t_full: np.ndarray, t_cens: np.ndarray) -> float:
    tau = float(theta[0])
    if tau <= 0:
        return np.inf
    ll_full = -np.log(tau) - t_full / tau   # log f(t)
    ll_cens = -t_cens / tau                  # log S(t)
    val = -(np.sum(ll_full) + np.sum(ll_cens))
    return val if np.isfinite(val) else np.inf


def nll_biexp(theta: Sequence[float], t_full: np.ndarray, t_cens: np.ndarray) -> float:
    pi, tau1, tau2 = float(theta[0]), float(theta[1]), float(theta[2])
    if not (0.0 < pi < 1.0) or tau1 <= 0 or tau2 <= 0:
        return np.inf
    f = pi / tau1 * np.exp(-t_full / tau1) + (1 - pi) / tau2 * np.exp(-t_full / tau2)
    s = pi * np.exp(-t_cens / tau1) + (1 - pi) * np.exp(-t_cens / tau2)
    val = -(np.sum(_safe_log(f)) + np.sum(_safe_log(s)))
    return val if np.isfinite(val) else np.inf


# ── Hessian (observed Fisher information) ──────────────────────────────────
def numerical_hessian(f, x: np.ndarray, eps_rel: float = 1e-4):
    n = len(x)
    eps = np.maximum(np.abs(x) * eps_rel, 1e-6)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy(); xpp[i] += eps[i]; xpp[j] += eps[j]
            xpm = x.copy(); xpm[i] += eps[i]; xpm[j] -= eps[j]
            xmp = x.copy(); xmp[i] -= eps[i]; xmp[j] += eps[j]
            xmm = x.copy(); xmm[i] -= eps[i]; xmm[j] -= eps[j]
            H[i, j] = H[j, i] = (
                (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps[i] * eps[j])
            )
    return H


def hessian_se(f, x: np.ndarray):
    try:
        H = numerical_hessian(f, x)
        Sigma = np.linalg.inv(H)
        return np.sqrt(np.maximum(np.diag(Sigma), 0))
    except np.linalg.LinAlgError:
        return np.full(len(x), np.nan)


# ── Information criteria ───────────────────────────────────────────────────
def aic(nll: float, k: int) -> float:
    return 2 * k + 2 * nll


def aicc(nll: float, k: int, n: int) -> float:
    if n - k - 1 <= 0:
        return np.nan
    return aic(nll, k) + (2 * k * (k + 1)) / (n - k - 1)


def bic(nll: float, k: int, n: int) -> float:
    return k * np.log(max(n, 1)) + 2 * nll


# ── Model-free metrics on events ───────────────────────────────────────────
def model_free_event_metrics(t_full: np.ndarray, horizons=HORIZONS) -> dict:
    """S(τ*) (Kaplan-Meier-like, no-censoring approximation) and ⟨τ⟩."""
    out = {}
    n = len(t_full)
    if n == 0:
        for h in horizons:
            out[f"S_{h:.0f}ns"] = np.nan
            out[f"RMST_{h:.0f}ns"] = np.nan
        out["mean_dwell_ns"] = np.nan
        out["median_dwell_ns"] = np.nan
        return out
    ts = np.sort(t_full)
    out["mean_dwell_ns"] = float(np.mean(ts))
    out["median_dwell_ns"] = float(np.median(ts))
    for h in horizons:
        out[f"S_{h:.0f}ns"] = float(np.sum(ts > h) / n)
        # RMST = ∫_0^h S(τ) dτ for empirical survival (step function)
        # Use sorted dwells: RMST = Σ τ_i I(τ_i ≤ h)/n + h * S(h)
        in_h = ts[ts <= h]
        rmst = float(np.sum(in_h) / n + h * (np.sum(ts > h) / n))
        out[f"RMST_{h:.0f}ns"] = rmst
    return out


# ── Fitting wrappers (MLE) ─────────────────────────────────────────────────
def fit_single_exp_mle(t_full: np.ndarray, t_cens: np.ndarray):
    """MLE for single-exp survival.  Closed-form: τ̂ = (Σt_full + Σt_cens) / N_full."""
    n_full, n_cens = len(t_full), len(t_cens)
    n_total = n_full + n_cens
    if n_full < 1:
        return None
    # MLE with right censoring: τ̂ = (Σt_all) / n_uncensored
    tau_hat = (t_full.sum() + t_cens.sum()) / n_full
    if tau_hat <= 0 or not np.isfinite(tau_hat):
        return None
    nll_min = nll_single([tau_hat], t_full, t_cens)
    se = hessian_se(lambda x: nll_single(x, t_full, t_cens), np.array([tau_hat]))
    k = 1
    return {
        "tau": tau_hat,
        "perr_tau": float(se[0]),
        "k_off": 1.0 / tau_hat,
        "nll": nll_min,
        "n_full": n_full,
        "n_cens": n_cens,
        "AIC": aic(nll_min, k),
        "AICc": aicc(nll_min, k, n_total),
        "BIC": bic(nll_min, k, n_total),
    }


def fit_bi_exp_mle(t_full: np.ndarray, t_cens: np.ndarray, n_starts: int = 8):
    """MLE for bi-exp mixture survival (3 free params: π, τ₁, τ₂)."""
    n_full, n_cens = len(t_full), len(t_cens)
    n_total = n_full + n_cens
    if n_full < 4:
        return None

    nll = lambda th: nll_biexp(th, t_full, t_cens)
    mean_t = float(np.mean(t_full))
    bounds = [(1e-3, 1.0 - 1e-3), (max(DT_NS / 4, 1e-3), 5 * mean_t),
              (max(DT_NS / 4, 1e-3), 200.0)]

    rng = np.random.default_rng(0)
    starts = []
    starts.append([0.5, max(DT_NS, mean_t / 5), 5 * mean_t])
    starts.append([0.7, mean_t / 4, 5 * mean_t])
    starts.append([0.9, DT_NS, 10.0])
    starts.append([0.95, DT_NS, 20.0])
    for _ in range(n_starts - 4):
        starts.append([
            rng.uniform(0.1, 0.95),
            rng.uniform(DT_NS, mean_t),
            rng.uniform(mean_t, 50.0),
        ])

    best = None
    for x0 in starts:
        try:
            r = optimize.minimize(
                nll, x0=x0, method="L-BFGS-B", bounds=bounds,
                options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 2000},
            )
        except Exception:
            continue
        if not r.success and not np.isfinite(r.fun):
            continue
        if best is None or r.fun < best.fun:
            best = r

    if best is None or not np.isfinite(best.fun):
        return None

    pi, tau1, tau2 = best.x
    # Canonicalise so τ₁ < τ₂ (fast first)
    if tau1 > tau2:
        pi, tau1, tau2 = 1 - pi, tau2, tau1

    se = hessian_se(nll, np.array([pi, tau1, tau2]))
    k = 3
    nll_min = best.fun
    # Effective rates
    k_off_fast = 1.0 / tau1
    k_off_slow = 1.0 / tau2
    # Mean dwell from mixture
    mean_dwell = pi * tau1 + (1 - pi) * tau2
    k_off_mean = 1.0 / mean_dwell
    return {
        "pi": pi, "tau_fast": tau1, "tau_slow": tau2,
        "alpha_fast": pi, "alpha_slow": 1 - pi,
        "perr_pi": float(se[0]),
        "perr_tau_fast": float(se[1]),
        "perr_tau_slow": float(se[2]),
        "k_off_fast": k_off_fast, "k_off_slow": k_off_slow,
        "k_off_mean": k_off_mean,
        "mean_dwell_ns": mean_dwell,
        "nll": nll_min,
        "n_full": n_full, "n_cens": n_cens,
        "AIC": aic(nll_min, k),
        "AICc": aicc(nll_min, k, n_total),
        "BIC": bic(nll_min, k, n_total),
        "t_half_fast": tau1 * np.log(2),
        "t_half_slow": tau2 * np.log(2),
    }


# ── R² calculated against empirical survival, for reporting only ───────────
def empirical_survival(t_full: np.ndarray, t_cens: np.ndarray):
    """Kaplan-Meier estimator with right-censoring."""
    if len(t_full) + len(t_cens) == 0:
        return np.array([0.0]), np.array([1.0])
    times = np.concatenate([t_full, t_cens])
    events = np.concatenate([np.ones_like(t_full), np.zeros_like(t_cens)]).astype(bool)
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    surv = 1.0
    n_at_risk = len(times)
    out_t = [0.0]; out_s = [1.0]
    for i, t in enumerate(times):
        if events[i]:
            surv *= (n_at_risk - 1) / n_at_risk if n_at_risk > 0 else 0
            out_t.append(t); out_s.append(surv)
        n_at_risk -= 1
    return np.asarray(out_t), np.asarray(out_s)


def r2_against_KM(model_S_fn, t_full, t_cens):
    t, s = empirical_survival(t_full, t_cens)
    yhat = model_S_fn(t)
    ss_tot = np.sum((s - s.mean()) ** 2)
    ss_res = np.sum((s - yhat) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ── k_on calculation ────────────────────────────────────────────────────────
def k_on_per_chain(arrivals_per_chain: dict) -> dict:
    """Returns {chain: k_on in M^-1 ns^-1}.

    k_on = (arrivals / T_total_ns) / [bulk]  [M⁻¹·ns⁻¹]
    """
    out = {}
    for ch, n in arrivals_per_chain.items():
        rate = n / T_TOTAL_NS                  # arrivals per ns
        out[ch] = rate / BULK_M if BULK_M > 0 else np.nan
    return out


# ── Output: per-row records for CSVs ───────────────────────────────────────
def collect_events_split(events: list[dict]):
    """Split events by chain → arrays for fitting."""
    by_chain = {ch: {"full": [], "cens": []} for ch in CHAINS}
    for e in events:
        ch = e["chain"]
        if e["left_cens"]:                # exclude left-censored from MLE
            continue
        bucket = "cens" if e["right_cens"] else "full"
        by_chain[ch][bucket].append(e["dwell_ns"])
    for ch in by_chain:
        by_chain[ch]["full"] = np.asarray(by_chain[ch]["full"], dtype=float)
        by_chain[ch]["cens"] = np.asarray(by_chain[ch]["cens"], dtype=float)
    # Pooled (across all 7 chains)
    full_all = np.concatenate([by_chain[c]["full"] for c in CHAINS])
    cens_all = np.concatenate([by_chain[c]["cens"] for c in CHAINS])
    return by_chain, full_all, cens_all


# ── CSV writers (residence-time style) ─────────────────────────────────────
SINGLE_COLS = [
    "buffer", "scope", "n_full", "n_cens", "n_total",
    "tau", "perr_tau", "k_off",
    "mean_dwell_ns", "median_dwell_ns",
    "S_1ns", "S_2ns", "S_5ns", "RMST_1ns", "RMST_2ns", "RMST_5ns",
    "R2_KM", "AIC", "AICc", "BIC", "nll",
]
BIEXP_COLS = [
    "buffer", "scope", "n_full", "n_cens", "n_total",
    "alpha_fast", "tau_fast", "perr_tau_fast",
    "alpha_slow", "tau_slow", "perr_tau_slow",
    "perr_pi",
    "k_off_fast", "k_off_slow", "k_off_mean", "mean_dwell_ns",
    "t_half_fast", "t_half_slow",
    "S_1ns", "S_2ns", "S_5ns", "RMST_1ns", "RMST_2ns", "RMST_5ns",
    "R2_KM", "AIC", "AICc", "BIC", "nll",
]


def fit_record(buf: str, scope: str, full: np.ndarray, cens: np.ndarray):
    """Run both single & bi fits + free metrics. Return (single_row, bi_row)."""
    n_full, n_cens = len(full), len(cens)
    free = model_free_event_metrics(full)

    s_fit = fit_single_exp_mle(full, cens)
    if s_fit is not None:
        s_S = lambda t, tau=s_fit["tau"]: np.exp(-t / tau)
        s_r2 = r2_against_KM(s_S, full, cens)
        single_row = {
            "buffer": buf, "scope": scope,
            "n_full": n_full, "n_cens": n_cens, "n_total": n_full + n_cens,
            "tau": s_fit["tau"], "perr_tau": s_fit["perr_tau"],
            "k_off": s_fit["k_off"],
            "mean_dwell_ns": free["mean_dwell_ns"],
            "median_dwell_ns": free["median_dwell_ns"],
            "S_1ns": free["S_1ns"], "S_2ns": free["S_2ns"], "S_5ns": free["S_5ns"],
            "RMST_1ns": free["RMST_1ns"], "RMST_2ns": free["RMST_2ns"],
            "RMST_5ns": free["RMST_5ns"],
            "R2_KM": s_r2,
            "AIC": s_fit["AIC"], "AICc": s_fit["AICc"], "BIC": s_fit["BIC"],
            "nll": s_fit["nll"],
        }
    else:
        single_row = {k: "" for k in SINGLE_COLS}
        single_row.update({"buffer": buf, "scope": scope,
                           "n_full": n_full, "n_cens": n_cens,
                           "n_total": n_full + n_cens})

    b_fit = fit_bi_exp_mle(full, cens)
    if b_fit is not None:
        pi = b_fit["pi"]; t1 = b_fit["tau_fast"]; t2 = b_fit["tau_slow"]
        b_S = lambda t: pi * np.exp(-t / t1) + (1 - pi) * np.exp(-t / t2)
        b_r2 = r2_against_KM(b_S, full, cens)
        bi_row = {
            "buffer": buf, "scope": scope,
            "n_full": n_full, "n_cens": n_cens, "n_total": n_full + n_cens,
            "alpha_fast": b_fit["alpha_fast"],
            "tau_fast": b_fit["tau_fast"],
            "perr_tau_fast": b_fit["perr_tau_fast"],
            "alpha_slow": b_fit["alpha_slow"],
            "tau_slow": b_fit["tau_slow"],
            "perr_tau_slow": b_fit["perr_tau_slow"],
            "perr_pi": b_fit["perr_pi"],
            "k_off_fast": b_fit["k_off_fast"],
            "k_off_slow": b_fit["k_off_slow"],
            "k_off_mean": b_fit["k_off_mean"],
            "mean_dwell_ns": b_fit["mean_dwell_ns"],
            "t_half_fast": b_fit["t_half_fast"],
            "t_half_slow": b_fit["t_half_slow"],
            "S_1ns": free["S_1ns"], "S_2ns": free["S_2ns"], "S_5ns": free["S_5ns"],
            "RMST_1ns": free["RMST_1ns"], "RMST_2ns": free["RMST_2ns"],
            "RMST_5ns": free["RMST_5ns"],
            "R2_KM": b_r2,
            "AIC": b_fit["AIC"], "AICc": b_fit["AICc"], "BIC": b_fit["BIC"],
            "nll": b_fit["nll"],
        }
    else:
        bi_row = {k: "" for k in BIEXP_COLS}
        bi_row.update({"buffer": buf, "scope": scope,
                       "n_full": n_full, "n_cens": n_cens,
                       "n_total": n_full + n_cens})
    return single_row, bi_row, s_fit, b_fit


def fmt_val(v):
    if v is None or v == "" or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_csv(rows: list[dict], path: str, cols: list[str]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt_val(r.get(c, "")) for c in cols])


# ── Per-buffer human-readable summary text ─────────────────────────────────
def write_summary_txt(buf: str, full: np.ndarray, cens: np.ndarray,
                      s_fit, b_fit, kon_per_chain: dict, arrivals: dict,
                      out_path: str):
    lines = []
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"DWELL-TIME KINETICS — buffer {buf}")
    lines.append(sep)
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"  T_total       = {T_TOTAL_NS:.2f} ns")
    lines.append(f"  dt(analysis)  = {DT_NS:.3f} ns/frame  (stride 10 on 0.01 ns trajectory)")
    lines.append(f"  Box           = {BOX_NM} nm cubic  →  V = {V_L*1e21:.2f} × 10⁻²¹ L")
    lines.append(f"  [ADP]_bulk    = {BULK_M*1000:.2f} mM  ({N_ADP} ADP / box)")
    lines.append(f"  Intermittency = {INTERMITTENCY} frames")
    lines.append("")
    lines.append("Event counts (pooled across 7 chains):")
    lines.append(f"  N (uncensored) = {len(full)}")
    lines.append(f"  N (right-censored, still bound at t=T) = {len(cens)}")
    lines.append(f"  Mean dwell (uncensored) = "
                 f"{(np.mean(full) if len(full) else np.nan):.4f} ns")
    lines.append(f"  Median dwell             = "
                 f"{(np.median(full) if len(full) else np.nan):.4f} ns")
    lines.append("")

    if s_fit is not None:
        lines.append("Single-exp MLE  f(t) = (1/τ) exp(-t/τ):")
        lines.append(f"  τ      = {s_fit['tau']:.4f} ns ± {s_fit['perr_tau']:.4f}")
        lines.append(f"  k_off  = 1/τ = {s_fit['k_off']:.4f} ns⁻¹  "
                     f"= {s_fit['k_off']*1e9:.3e} s⁻¹")
        lines.append(f"  −logL  = {s_fit['nll']:.4f}")
        lines.append(f"  AIC    = {s_fit['AIC']:.2f}    "
                     f"AICc = {s_fit['AICc']:.2f}    "
                     f"BIC = {s_fit['BIC']:.2f}")
    lines.append("")

    if b_fit is not None:
        lines.append("Bi-exp MLE  f(t) = (π/τ₁) exp(-t/τ₁) + ((1-π)/τ₂) exp(-t/τ₂):")
        lines.append(f"  π       = {b_fit['pi']:.4f} ± {b_fit['perr_pi']:.4f}    "
                     f"(amplitude of fast component)")
        lines.append(f"  τ_fast  = {b_fit['tau_fast']:.4f} ns ± {b_fit['perr_tau_fast']:.4f}")
        lines.append(f"  τ_slow  = {b_fit['tau_slow']:.4f} ns ± {b_fit['perr_tau_slow']:.4f}")
        lines.append(f"  k_off_fast = 1/τ_fast = {b_fit['k_off_fast']:.4f} ns⁻¹  "
                     f"= {b_fit['k_off_fast']*1e9:.3e} s⁻¹")
        lines.append(f"  k_off_slow = 1/τ_slow = {b_fit['k_off_slow']:.4f} ns⁻¹  "
                     f"= {b_fit['k_off_slow']*1e9:.3e} s⁻¹")
        lines.append(f"  ⟨τ⟩ (mixture) = π·τ_fast + (1-π)·τ_slow = "
                     f"{b_fit['mean_dwell_ns']:.4f} ns")
        lines.append(f"  k_off_mean = 1/⟨τ⟩ = {b_fit['k_off_mean']:.4f} ns⁻¹")
        lines.append(f"  t½_fast = {b_fit['t_half_fast']:.4f} ns,  "
                     f"t½_slow = {b_fit['t_half_slow']:.4f} ns")
        lines.append(f"  −logL = {b_fit['nll']:.4f}")
        lines.append(f"  AIC = {b_fit['AIC']:.2f}    "
                     f"AICc = {b_fit['AICc']:.2f}    "
                     f"BIC = {b_fit['BIC']:.2f}")
    lines.append("")

    if s_fit is not None and b_fit is not None:
        d_aic = s_fit["AIC"] - b_fit["AIC"]
        d_bic = s_fit["BIC"] - b_fit["BIC"]
        lines.append("Model selection (single vs bi-exp):")
        lines.append(f"  ΔAIC = AIC_single − AIC_bi = {d_aic:.2f}  "
                     f"({'bi-exp preferred' if d_aic > 2 else 'inconclusive' if d_aic > -2 else 'single preferred'})")
        lines.append(f"  ΔBIC = BIC_single − BIC_bi = {d_bic:.2f}  "
                     f"({'bi-exp preferred' if d_bic > 2 else 'inconclusive' if d_bic > -2 else 'single preferred'})")
        lines.append("")

    # k_on per chain
    lines.append("k_on per chain  (M⁻¹·ns⁻¹  →  M⁻¹·s⁻¹):")
    lines.append(f"  k_on = (N_arrivals / T_total) / [ADP]_bulk")
    lines.append("  chain   N_arrivals    k_on (M⁻¹ns⁻¹)    k_on (M⁻¹s⁻¹)")
    chain_avg_kon = []
    for ch in CHAINS:
        n_arr = arrivals.get(ch, 0)
        kon = kon_per_chain.get(ch, np.nan)
        chain_avg_kon.append(kon)
        lines.append(f"  {ch}       {n_arr:>5d}        "
                     f"{kon:>9.4f}         {kon*1e9:.3e}")
    kon_total = np.nanmean(chain_avg_kon)
    n_arr_total = sum(arrivals.values())
    lines.append(f"  ─── chain-averaged: k_on = {kon_total:.4f} M⁻¹·ns⁻¹ "
                 f"= {kon_total*1e9:.3e} M⁻¹·s⁻¹")
    lines.append(f"  ─── total arrivals:  N = {n_arr_total}")
    lines.append("")

    # K_d
    if b_fit is not None and kon_total > 0:
        kd_fast = b_fit["k_off_fast"] / kon_total
        kd_slow = b_fit["k_off_slow"] / kon_total
        lines.append("Equilibrium dissociation constant  K_d = k_off / k_on:")
        lines.append(f"  K_d (fast component) = {kd_fast:.4g} M  "
                     f"({kd_fast*1000:.2f} mM)")
        lines.append(f"  K_d (slow component) = {kd_slow:.4g} M  "
                     f"({kd_slow*1000:.4f} mM)")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── k_on / k_off table per chain × buffer ──────────────────────────────────
def write_kon_koff_summary(per_buffer_data: dict, path: str):
    """Write explicit per-chain k_on, k_off table for both buffers."""
    cols = ["buffer", "chain", "N_events", "N_full", "N_cens", "N_arrivals",
            "k_on_M_ns", "k_on_M_s",
            "k_off_single", "k_off_fast", "k_off_slow", "k_off_mean",
            "tau_single", "tau_fast", "tau_slow",
            "K_d_slow_mM", "K_d_fast_mM"]
    rows = []
    for buf, data in per_buffer_data.items():
        by_chain = data["by_chain"]
        kons = data["kon_per_chain"]
        arrivals = data["arrivals"]
        for ch in CHAINS:
            full = by_chain[ch]["full"]
            cens = by_chain[ch]["cens"]
            n_full, n_cens = len(full), len(cens)
            n_total = n_full + n_cens
            n_arr = arrivals.get(ch, 0)
            kon = kons.get(ch, np.nan)
            row = {"buffer": buf, "chain": ch,
                   "N_events": n_total, "N_full": n_full, "N_cens": n_cens,
                   "N_arrivals": n_arr,
                   "k_on_M_ns": kon, "k_on_M_s": kon * 1e9 if np.isfinite(kon) else np.nan}
            if n_full >= 4:
                s_fit = fit_single_exp_mle(full, cens)
                b_fit = fit_bi_exp_mle(full, cens)
                if s_fit:
                    row["tau_single"] = s_fit["tau"]
                    row["k_off_single"] = s_fit["k_off"]
                if b_fit:
                    row["tau_fast"] = b_fit["tau_fast"]
                    row["tau_slow"] = b_fit["tau_slow"]
                    row["k_off_fast"] = b_fit["k_off_fast"]
                    row["k_off_slow"] = b_fit["k_off_slow"]
                    row["k_off_mean"] = b_fit["k_off_mean"]
                    if kon > 0 and np.isfinite(kon):
                        row["K_d_fast_mM"] = (b_fit["k_off_fast"] / kon) * 1000
                        row["K_d_slow_mM"] = (b_fit["k_off_slow"] / kon) * 1000
            elif n_full >= 1:
                s_fit = fit_single_exp_mle(full, cens)
                if s_fit:
                    row["tau_single"] = s_fit["tau"]
                    row["k_off_single"] = s_fit["k_off"]
            rows.append(row)
    write_csv(rows, path, cols)


# ── Plot ───────────────────────────────────────────────────────────────────
def make_plots(per_buffer_data: dict, plot_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    colors = {"EDDA": "#d95f02", "AMAC": "#1f78b4"}

    # Per-buffer figure with 4 panels.
    for buf, data in per_buffer_data.items():
        full = data["full_all"]; cens = data["cens_all"]
        s_fit = data["pooled_single"]
        b_fit = data["pooled_bi"]
        if len(full) == 0:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        # Panel A: PDF histogram (log-log) + bi-exp PDF.
        # Use log-spaced bins for shape, plot density per-decade so heights
        # are visually comparable and don't run to 1e-50.
        ax = axes[0, 0]
        tmin = max(DT_NS, full.min())
        tmax = full.max() * 1.1
        bins = np.logspace(np.log10(tmin), np.log10(tmax), 25)
        counts, edges = np.histogram(full, bins=bins)
        widths = np.diff(edges)
        # density = counts / (N * width)  — true PDF; clip empty bins for log y
        density = counts / (len(full) * widths)
        density = np.where(counts > 0, density, np.nan)
        centers = (edges[:-1] * edges[1:]) ** 0.5
        ax.bar(centers, density, width=widths, align="center",
               color=colors[buf], alpha=0.55, edgecolor="white",
               label=f"data PDF  (N={len(full)})")
        if b_fit:
            tt = np.logspace(np.log10(tmin), np.log10(tmax), 300)
            pdf = (b_fit["pi"] / b_fit["tau_fast"] * np.exp(-tt / b_fit["tau_fast"])
                   + (1 - b_fit["pi"]) / b_fit["tau_slow"] * np.exp(-tt / b_fit["tau_slow"]))
            ax.plot(tt, pdf, color="black", lw=1.5,
                    label=(f"bi-exp MLE\n"
                           f"  π={b_fit['pi']:.3f}±{b_fit['perr_pi']:.3f}\n"
                           f"  τ_fast={b_fit['tau_fast']:.2f}±{b_fit['perr_tau_fast']:.2f} ns\n"
                           f"  τ_slow={b_fit['tau_slow']:.2f}±{b_fit['perr_tau_slow']:.2f} ns"))
        if s_fit:
            tt = np.logspace(np.log10(tmin), np.log10(tmax), 300)
            ax.plot(tt, (1 / s_fit["tau"]) * np.exp(-tt / s_fit["tau"]),
                    "--", color="gray", lw=1.0,
                    label=f"single-exp MLE\n  τ={s_fit['tau']:.2f}±{s_fit['perr_tau']:.2f} ns")
        ax.set_xscale("log"); ax.set_yscale("log")
        # ylim: clip to the lowest non-empty bin so empty cells don't drag axis.
        valid = density[~np.isnan(density)]
        if len(valid) > 0:
            ymin = max(valid.min() * 0.5, 1e-6)
            ymax = valid.max() * 2.0
            ax.set_ylim(ymin, ymax)
        ax.set_xlim(tmin * 0.9, tmax)
        ax.set_xlabel("dwell τ (ns)"); ax.set_ylabel("density f(τ)")
        ax.set_title(f"{buf} — dwell PDF (log-log)")
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")
        ax.grid(True, alpha=0.3, which="both")

        # Panel B: KM survival + bi-exp + single-exp
        ax = axes[0, 1]
        t_e, s_e = empirical_survival(full, cens)
        ax.step(t_e, s_e, where="post", color=colors[buf], lw=1.6,
                label="Kaplan-Meier")
        if b_fit:
            tt = np.linspace(0, t_e.max() * 1.05, 300)
            s_bi = b_fit["pi"] * np.exp(-tt / b_fit["tau_fast"]) + \
                   (1 - b_fit["pi"]) * np.exp(-tt / b_fit["tau_slow"])
            ax.plot(tt, s_bi, "k-", lw=1.4, label="bi-exp")
        if s_fit:
            tt = np.linspace(0, t_e.max() * 1.05, 300)
            ax.plot(tt, np.exp(-tt / s_fit["tau"]), "--", color="gray",
                    lw=1.0, label="single-exp")
        ax.set_xlabel("τ (ns)"); ax.set_ylabel("S(τ)")
        ax.set_yscale("log"); ax.set_ylim(1e-3, 1.05)
        ax.set_title(f"{buf} — survival (log)")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3, which="both")

        # Panel C: cumulative arrivals (events / time, per chain bars)
        ax = axes[1, 0]
        chains = CHAINS
        n_arrivals = [data["arrivals"].get(ch, 0) for ch in chains]
        ax.bar(chains, n_arrivals, color=colors[buf], alpha=0.75, edgecolor="white")
        for i, n in enumerate(n_arrivals):
            ax.text(i, n + max(n_arrivals) * 0.01, f"{n}", ha="center",
                    va="bottom", fontsize=8)
        ax.set_xlabel("chain"); ax.set_ylabel("N_arrivals (500 ns)")
        ax.set_title(f"{buf} — arrivals per chain")
        ax.grid(True, alpha=0.3, axis="y")

        # Panel D: per-chain k_on bars
        ax = axes[1, 1]
        kons = [data["kon_per_chain"].get(ch, np.nan) for ch in chains]
        ax.bar(chains, kons, color=colors[buf], alpha=0.75, edgecolor="white")
        ax.set_xlabel("chain"); ax.set_ylabel("k_on  (M⁻¹·ns⁻¹)")
        ax.set_title(f"{buf} — apparent k_on per chain")
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Dwell-time kinetics — {buf}  "
                     f"(N_full={len(full)}, N_cens={len(cens)})", y=0.995)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"dwell_{buf}.svg"),
                    facecolor="white")
        fig.savefig(os.path.join(plot_dir, f"dwell_{buf}.png"),
                    dpi=140, facecolor="white")
        plt.close(fig)

    # Combined figure: overlay of both KM survival + bi-exp fits
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for buf, data in per_buffer_data.items():
        full = data["full_all"]; cens = data["cens_all"]
        b_fit = data["pooled_bi"]
        if len(full) == 0:
            continue
        t_e, s_e = empirical_survival(full, cens)
        for ax, scale in zip(axes, ("linear", "log")):
            ax.step(t_e, s_e, where="post", color=colors[buf], lw=1.6, alpha=0.9,
                    label=f"{buf} KM  (N={len(full)})")
            if b_fit:
                tt = np.linspace(0, t_e.max() * 1.05, 300)
                s_bi = b_fit["pi"] * np.exp(-tt / b_fit["tau_fast"]) + \
                       (1 - b_fit["pi"]) * np.exp(-tt / b_fit["tau_slow"])
                ax.plot(tt, s_bi, "--", color=colors[buf], lw=1.0,
                        label=(f"{buf} bi-exp:  π={b_fit['pi']:.2f}, "
                               f"τ₁={b_fit['tau_fast']:.2f}, "
                               f"τ₂={b_fit['tau_slow']:.1f} ns"))
    for ax, scale in zip(axes, ("linear", "log")):
        ax.set_yscale(scale)
        ax.set_xlabel("dwell τ (ns)")
        ax.set_xlim(0, 30)
        ax.set_ylabel("S(τ)" + (" (log)" if scale == "log" else ""))
        if scale == "log":
            ax.set_ylim(1e-3, 1.05)
        else:
            ax.set_ylim(0, 1.05)
        ax.set_title(f"survival ({scale})")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle("Dwell-time kinetics — EDDA vs AMAC")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "dwell_combined.svg"), facecolor="white")
    fig.savefig(os.path.join(plot_dir, "dwell_combined.png"), dpi=140, facecolor="white")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────
def write_events_csv(events, path):
    keys = ["chain", "adp_resid", "t_start_ns", "dwell_ns", "left_cens", "right_cens"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for e in events:
            w.writerow(e)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    single_rows: list[dict] = []
    biexp_rows: list[dict] = []
    per_buffer_data: dict = {}

    for buf in BUFFERS:
        events, arrivals = collect_events(buf)
        write_events_csv(events, os.path.join(OUT_DIR, f"dwell_events_{buf}.csv"))

        by_chain, full_all, cens_all = collect_events_split(events)
        kon_per_chain = k_on_per_chain(arrivals)

        # Pooled fit (across chains)
        s_pool, b_pool, s_fit_obj, b_fit_obj = fit_record(buf, "pooled", full_all, cens_all)
        single_rows.append(s_pool); biexp_rows.append(b_pool)

        # Per-chain fits
        for ch in CHAINS:
            full = by_chain[ch]["full"]; cens = by_chain[ch]["cens"]
            s_row, b_row, _, _ = fit_record(buf, f"chain_{ch}", full, cens)
            single_rows.append(s_row); biexp_rows.append(b_row)

        per_buffer_data[buf] = {
            "events": events,
            "by_chain": by_chain,
            "full_all": full_all, "cens_all": cens_all,
            "arrivals": arrivals,
            "kon_per_chain": kon_per_chain,
            "pooled_single": s_fit_obj,
            "pooled_bi": b_fit_obj,
        }

        # Summary text
        write_summary_txt(buf, full_all, cens_all,
                          s_fit_obj, b_fit_obj,
                          kon_per_chain, arrivals,
                          os.path.join(OUT_DIR, f"summary_{buf}.txt"))

    # CSVs
    write_csv(single_rows,
              os.path.join(OUT_DIR, "single_exp_fitting_results.csv"),
              SINGLE_COLS)
    write_csv(biexp_rows,
              os.path.join(OUT_DIR, "bi_exp_fitting_results.csv"),
              BIEXP_COLS)
    write_kon_koff_summary(per_buffer_data,
                           os.path.join(OUT_DIR, "kon_koff_summary.csv"))

    # Plots
    make_plots(per_buffer_data, PLOT_DIR)

    # Console summary (compact)
    print("=" * 60)
    print("Dwell-time kinetics — pooled MLE results")
    print("=" * 60)
    for buf in BUFFERS:
        d = per_buffer_data[buf]
        b = d["pooled_bi"]; s = d["pooled_single"]
        n_full = len(d["full_all"]); n_cens = len(d["cens_all"])
        n_arr = sum(d["arrivals"].values())
        kon_avg = np.nanmean(list(d["kon_per_chain"].values()))
        print(f"\n── {buf} ──")
        print(f"  N_events: full={n_full}  cens={n_cens}  arrivals={n_arr}")
        if s:
            print(f"  Single MLE:  τ = {s['tau']:.4f} ± {s['perr_tau']:.4f} ns "
                  f"(k_off = {s['k_off']:.4f} ns⁻¹)")
        if b:
            print(f"  Bi-exp MLE:  π = {b['pi']:.3f} ± {b['perr_pi']:.3f}, "
                  f"τ_fast = {b['tau_fast']:.3f} ± {b['perr_tau_fast']:.3f}, "
                  f"τ_slow = {b['tau_slow']:.3f} ± {b['perr_tau_slow']:.3f} ns")
            print(f"  k_off_slow = {b['k_off_slow']:.4f} ns⁻¹  "
                  f"= {b['k_off_slow']*1e9:.3e} s⁻¹")
        if s and b:
            d_aic = s["AIC"] - b["AIC"]
            print(f"  ΔAIC (single − bi): {d_aic:.2f}  "
                  f"({'bi-exp preferred' if d_aic > 2 else 'inconclusive' if d_aic > -2 else 'single preferred'})")
        print(f"  k_on (chain-avg): {kon_avg:.4f} M⁻¹·ns⁻¹  "
              f"= {kon_avg*1e9:.3e} M⁻¹·s⁻¹")
        if b:
            kd_fast = b["k_off_fast"] / kon_avg if kon_avg > 0 else np.nan
            kd_slow = b["k_off_slow"] / kon_avg if kon_avg > 0 else np.nan
            print(f"  K_d (fast) = {kd_fast*1000:.2f} mM,  "
                  f"K_d (slow) = {kd_slow*1000:.4f} mM")

    print(f"\nWrote: {OUT_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
