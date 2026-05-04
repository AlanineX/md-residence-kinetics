#!/usr/bin/env python3
"""Dwell-time kinetics — single-species, event-based view.

Alternative to the population-decay (bi-exp on n(t)) approach. Reads the
T1 _intermediate/contacts/chain_X.npz CSR-style files for each chain and:

  1. Reconstructs per-ADP-molecule binary occupancy time series.
  2. Extracts dwell intervals (consecutive bound runs) per molecule.
  3. Pools dwells across all 7 chains per buffer.
  4. Fits the survival function 1-CDF(τ) with single-exp and bi-exp.
  5. Counts arrivals (0→1 transitions) → k_on per chain in M^-1 ns^-1.

Outputs (results_v3/dwell_kinetics/):
  dwell_events_<buf>.csv      — every event: chain, adp_resid, t_start_ns, dwell_ns, censored
  dwell_summary_<buf>.csv     — per-chain: N_events, ⟨τ⟩, k_on, k_off
  dwell_kinetics.png          — survival curves (EDDA vs AMAC) + bi-exp fits
  dwell_kinetics_summary.txt  — overall numbers and fit parameters

Usage:
    python dwell_time_kinetics.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

# ── Constants ──────────────────────────────────────────────────────────────
RES = "/home/alan/working/groel_new/results_v3"
DT_NS = 0.10                # 10× stride on 0.01 ns trajectory
N_FRAMES = 5001             # frames in stored contacts npz
T_TOTAL_NS = (N_FRAMES - 1) * DT_NS  # 500.0 ns
N_ADP = 300                 # bulk ADP count (identical in both buffers)
BOX_NM = 19.1               # cubic edge
# 1 nm^3 = 1e-24 L  (1 nm = 1e-7 cm → 1e-21 cm^3 → /1000 → 1e-24 L)
V_L = BOX_NM ** 3 * 1e-24   # ≈ 6.97e-21 L
N_AVO = 6.02214076e23
BULK_M = N_ADP / (N_AVO * V_L)  # ≈ 0.0715 M = 71.5 mM
CHAINS = list("ABCDEFG")
BUFFERS = ("EDDA", "AMAC")
INTERMITTENCY = 0           # frames a molecule is allowed to leave & come back
                            # without breaking a dwell event (matches T1 setting)
OUT_DIR = os.path.join(RES, "dwell_kinetics")


# ── Per-molecule occupancy reconstruction ──────────────────────────────────
def load_occupancy(npz_path: str) -> dict[int, np.ndarray]:
    """Return {adp_resid: binary array of length N_FRAMES}."""
    d = np.load(npz_path, allow_pickle=True)
    fi = d["frame_indices"]
    off = d["offsets"]
    ri = d["resindices"]
    if len(ri) == 0:
        return {}

    # Build per-molecule frame-index lists.
    by_mol: dict[int, list[int]] = {}
    for i, frame in enumerate(fi):
        for resid in ri[off[i]:off[i + 1]]:
            by_mol.setdefault(int(resid), []).append(int(frame))

    # Frame indices in the file correspond to the original 50001-frame
    # trajectory (0, 10, 20, ...). Map to local index 0..5000.
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
        # Close short gaps of zeros (length ≤ intermittency).
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
    """Return (events, n_arrivals_per_chain). events is a list of dicts."""
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
                # Right-censored = event still bound at last frame.
                # Left-censored  = event was already bound at frame 0.
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
                    n_arr += 1   # only count *new* arrivals for k_on
        arrivals[ch] = n_arr
    return events, arrivals


# ── Survival-function fitting ──────────────────────────────────────────────
def survival(dwells: np.ndarray):
    """Empirical S(τ) = P(T > τ), evaluated at the unique dwell times."""
    d = np.sort(dwells)
    n = len(d)
    # at τ just below d[i], survival = (n - i) / n
    tau = np.concatenate([[0.0], d])
    s = np.concatenate([[1.0], (n - 1 - np.arange(n)) / n])
    return tau, s


def single_exp(t, tau): return np.exp(-t / tau)
def bi_exp(t, A, tau1, tau2): return A * np.exp(-t / tau1) + (1 - A) * np.exp(-t / tau2)


def fit_survival(tau: np.ndarray, s: np.ndarray):
    out = {}
    # single-exp
    try:
        p, _ = curve_fit(single_exp, tau, s, p0=[max(np.mean(tau), 1.0)],
                         bounds=([1e-3], [1e4]), maxfev=10000)
        ss_res = np.sum((s - single_exp(tau, *p)) ** 2)
        ss_tot = np.sum((s - s.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        out["single"] = {"tau": p[0], "r2": r2}
    except Exception as e:
        out["single"] = {"err": str(e)}
    # bi-exp
    try:
        p, _ = curve_fit(bi_exp, tau, s, p0=[0.5, 1.0, 20.0],
                         bounds=([0, 1e-3, 1e-3], [1, 1e4, 1e4]), maxfev=20000)
        if p[1] > p[2]:  # canonicalize: tau1 = fast (smaller)
            p = [1 - p[0], p[2], p[1]]
        ss_res = np.sum((s - bi_exp(tau, *p)) ** 2)
        ss_tot = np.sum((s - s.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        out["bi"] = {"A": p[0], "tau_fast": p[1], "tau_slow": p[2], "r2": r2}
    except Exception as e:
        out["bi"] = {"err": str(e)}
    return out


# ── Outputs ────────────────────────────────────────────────────────────────
def write_events_csv(events, path):
    keys = ["chain", "adp_resid", "t_start_ns", "dwell_ns", "left_cens", "right_cens"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for e in events:
            w.writerow(e)


def per_chain_summary(events, arrivals):
    """Return dict {chain: {N, mean_dwell, k_off, k_on}}."""
    out = {}
    for ch in CHAINS:
        ev = [e for e in events if e["chain"] == ch]
        n_arr = arrivals.get(ch, 0)
        if not ev:
            out[ch] = {"N": 0, "mean_dwell_ns": np.nan,
                       "k_off_per_ns": np.nan, "k_on_M_ns": np.nan}
            continue
        # Use only fully observed (uncensored) events for τ estimate.
        full = np.array([e["dwell_ns"] for e in ev
                         if not (e["left_cens"] or e["right_cens"])])
        mean_dw = float(np.mean(full)) if len(full) else np.nan
        k_off = 1.0 / mean_dw if mean_dw and mean_dw > 0 else np.nan
        # k_on: arrivals per ns per [bulk] per chain.
        k_on = (n_arr / T_TOTAL_NS) / BULK_M    # M^-1 ns^-1
        out[ch] = {"N": len(ev), "N_full": int(len(full)), "N_arrivals": n_arr,
                   "mean_dwell_ns": mean_dw,
                   "k_off_per_ns": k_off,
                   "k_on_M_ns": k_on}
    return out


def write_per_chain_csv(summary, path):
    keys = ["chain", "N", "N_full", "N_arrivals",
            "mean_dwell_ns", "k_off_per_ns", "k_on_M_ns"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for ch, d in summary.items():
            w.writerow([ch] + [d.get(k, "") for k in keys[1:]])


def _bi_exp_pdf(t, A, tau1, tau2):
    """Density form: -dS/dt for bi-exp survival."""
    return A / tau1 * np.exp(-t / tau1) + (1 - A) / tau2 * np.exp(-t / tau2)


def make_plot(per_buffer_events, per_buffer_fits, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"EDDA": "#d95f02", "AMAC": "#1f78b4"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))

    # Top row: per-buffer dwell-time histogram + bi-exp PDF overlay (log y).
    for col, buf in enumerate(("EDDA", "AMAC")):
        ax = axes[0, col]
        events = per_buffer_events[buf]
        full = np.array([e["dwell_ns"] for e in events
                         if not (e["left_cens"] or e["right_cens"])])
        if len(full) == 0:
            ax.text(0.5, 0.5, "no events", ha="center", va="center",
                    transform=ax.transAxes, color=colors[buf])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{buf} — N=0")
            continue

        # log-spaced bins to handle wide dynamic range.
        tmin = max(DT_NS / 2, full.min() * 0.9)
        tmax = full.max() * 1.1
        bins = np.logspace(np.log10(tmin), np.log10(tmax), 30)
        ax.hist(full, bins=bins, density=True, color=colors[buf],
                alpha=0.55, edgecolor="white",
                label=f"empirical PDF  (N={len(full)})")

        fit = per_buffer_fits[buf]
        if "bi" in fit and "tau_fast" in fit["bi"]:
            p = fit["bi"]
            tt = np.logspace(np.log10(tmin), np.log10(tmax), 250)
            ax.plot(tt, _bi_exp_pdf(tt, p["A"], p["tau_fast"], p["tau_slow"]),
                    color="black", lw=1.4,
                    label=(f"bi-exp PDF\n"
                           f"  A={p['A']:.2f}\n"
                           f"  τ_fast={p['tau_fast']:.2f} ns\n"
                           f"  τ_slow={p['tau_slow']:.2f} ns\n"
                           f"  R²={p['r2']:.3f}"))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("dwell time τ (ns)")
        ax.set_ylabel("density  P(τ)")
        ax.set_title(f"{buf} — dwell-time histogram")
        ax.legend(fontsize=8, loc="lower left", frameon=False)
        ax.grid(True, alpha=0.3, which="both")

    # Bottom row: survival S(τ) — linear (left) and log (right), no sharey.
    for col, scale in enumerate(("linear", "log")):
        ax = axes[1, col]
        for buf, events in per_buffer_events.items():
            full = np.array([e["dwell_ns"] for e in events
                             if not (e["left_cens"] or e["right_cens"])])
            if len(full) == 0:
                continue
            tau, s = survival(full)
            ax.step(tau, s, where="post", color=colors[buf],
                    lw=1.6, alpha=0.9, label=f"{buf}  (N={len(full)})")
            fit = per_buffer_fits[buf]
            if "tau_fast" in fit.get("bi", {}):
                p = fit["bi"]
                tt = np.linspace(0, tau.max(), 250)
                ax.plot(tt, bi_exp(tt, p["A"], p["tau_fast"], p["tau_slow"]),
                        ls="--", color=colors[buf], lw=1.0,
                        label=f"{buf} bi-exp fit (R²={p['r2']:.3f})")
        ax.set_xlabel("dwell time τ (ns)")
        ax.set_xlim(0, 30)         # focus on relevant range
        ax.set_yscale(scale)
        if scale == "log":
            ax.set_ylim(1e-3, 1.05)
            ax.set_ylabel("S(τ) = P(T > τ)  (log)")
        else:
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("S(τ) = P(T > τ)")
        ax.set_title(f"survival ({scale})")
        ax.legend(fontsize=8.5, loc="upper right", frameon=False)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Single-species dwell kinetics — EDDA vs AMAC", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, facecolor="white")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    per_buffer_events = {}
    per_buffer_fits = {}
    summary_lines = ["Dwell-time kinetics summary",
                     "=" * 60,
                     f"DT = {DT_NS} ns/frame, T_total = {T_TOTAL_NS:.1f} ns",
                     f"Box: {BOX_NM} nm cubic, [ADP]_bulk = {BULK_M*1000:.2f} mM",
                     f"Intermittency: {INTERMITTENCY} frames",
                     ""]

    for buf in BUFFERS:
        events, arrivals = collect_events(buf)
        per_buffer_events[buf] = events

        # Per-chain summary CSV
        chain_sum = per_chain_summary(events, arrivals)
        write_per_chain_csv(chain_sum,
                            os.path.join(OUT_DIR, f"dwell_summary_{buf}.csv"))
        write_events_csv(events,
                         os.path.join(OUT_DIR, f"dwell_events_{buf}.csv"))

        # Pooled fit (uncensored only)
        full = np.array([e["dwell_ns"] for e in events
                         if not (e["left_cens"] or e["right_cens"])])
        n_total = len(events)
        n_full = len(full)
        n_arrivals_total = sum(arrivals.values())

        if n_full > 0:
            tau, s = survival(full)
            fit = fit_survival(tau, s)
        else:
            fit = {"single": {"err": "no events"}, "bi": {"err": "no events"}}
        per_buffer_fits[buf] = fit

        summary_lines.append(f"── {buf} ──")
        summary_lines.append(f"  Events total: {n_total}  (uncensored: {n_full})")
        summary_lines.append(f"  Arrivals (sum across 7 chains): {n_arrivals_total}")
        summary_lines.append(f"  ⟨dwell⟩ (uncensored): "
                             f"{np.mean(full) if n_full else np.nan:.3f} ns")
        if "tau" in fit.get("single", {}):
            p = fit["single"]
            summary_lines.append(
                f"  Single-exp:  τ = {p['tau']:.3f} ns,  R² = {p['r2']:.3f}")
        if "tau_fast" in fit.get("bi", {}):
            p = fit["bi"]
            summary_lines.append(
                f"  Bi-exp:      A = {p['A']:.3f}, "
                f"τ_fast = {p['tau_fast']:.3f} ns, "
                f"τ_slow = {p['tau_slow']:.3f} ns, "
                f"R² = {p['r2']:.3f}")

        # Buffer-level k_on (sum of arrivals over 7 chains, per chain)
        # k_on chain-avg = arrivals / (7 chains × T_total × [bulk])
        k_on = (n_arrivals_total / 7.0) / T_TOTAL_NS / BULK_M
        summary_lines.append(
            f"  k_on (chain-avg): {k_on:.4f} M⁻¹·ns⁻¹  "
            f"= {k_on * 1e9:.2e} M⁻¹·s⁻¹")
        summary_lines.append("")

    # K_d = k_off / k_on for both bi-exp components
    summary_lines.append("── Affinity (k_off / k_on) ──")
    for buf in BUFFERS:
        fit = per_buffer_fits[buf]
        events, arrivals = per_buffer_events[buf], None
        n_arr = sum(a for ch, a in (
            (ch, sum(1 for e in events if e["chain"] == ch and not e["left_cens"]))
            for ch in CHAINS))
        k_on = (n_arr / 7.0) / T_TOTAL_NS / BULK_M
        if "tau_fast" in fit.get("bi", {}):
            p = fit["bi"]
            kd_fast = (1 / p["tau_fast"]) / k_on if k_on > 0 else np.nan
            kd_slow = (1 / p["tau_slow"]) / k_on if k_on > 0 else np.nan
            summary_lines.append(
                f"  {buf}:  K_d_fast ≈ {kd_fast:.3f} M  ({kd_fast*1000:.1f} mM),  "
                f"K_d_slow ≈ {kd_slow:.4f} M  ({kd_slow*1000:.2f} mM)")

    summary_path = os.path.join(OUT_DIR, "dwell_kinetics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    make_plot(per_buffer_events, per_buffer_fits,
              os.path.join(OUT_DIR, "dwell_kinetics.png"))
    print("\n".join(summary_lines))
    print(f"\nWrote: {OUT_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
