#!/usr/bin/env python3
"""Side-by-side demonstration of the two residence-time methods.

Pedagogical script — runs Method 1 (population SP autocorrelation) and
Method 2 (event-based dwell MLE) on the SAME real input (EDDA chain F)
and prints every intermediate value so the difference between the two
philosophies is concrete.

Pairs with docs/dwell_kinetics_methods_note.md.

Usage:
    python demo_two_methods.py              # default = EDDA chain F
    python demo_two_methods.py AMAC A       # any buffer × chain
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dwell_time_kinetics import (
    BULK_M, DT_NS, INTERMITTENCY, N_FRAMES, T_TOTAL_NS,
    fit_bi_exp_mle, fit_single_exp_mle, load_occupancy, runs_to_events,
)


RES_ROOT = "/home/alan/working/groel_new/results_v3"


def section(title):
    print("\n" + "═" * 70)
    print(title)
    print("═" * 70)


def main():
    buf = sys.argv[1] if len(sys.argv) > 1 else "EDDA"
    chain = sys.argv[2] if len(sys.argv) > 2 else "F"
    contact_path = os.path.join(
        RES_ROOT, f"T1_per_chain_pocket_{buf}",
        "_intermediate", "contacts", f"chain_{chain}.npz",
    )
    if not os.path.exists(contact_path):
        sys.exit(f"missing: {contact_path}")

    section(f"Setup — {buf} chain {chain}, 500 ns trajectory")
    print(f"  dt(analysis) = {DT_NS} ns  (stride 10 on 0.01 ns trajectory)")
    print(f"  T_total      = {T_TOTAL_NS} ns")
    print(f"  N_frames     = {N_FRAMES}")
    print(f"  [ADP]_bulk   = {BULK_M*1000:.2f} mM  (300 ADP / 19.1³ nm³ box)")
    print(f"  contact file = {contact_path}")

    # ── Shared step: build per-molecule occupancy series ───────────────────
    section("Shared step — per-molecule occupancy reconstruction")
    occ = load_occupancy(contact_path)
    print(f"  Unique ADP molecules ever in contact: {len(occ)}")
    if not occ:
        sys.exit("no events — exiting")
    for resid, arr in list(occ.items())[:3]:
        n_bound_frames = int(arr.sum())
        print(f"    resid={resid:5d}  bound for {n_bound_frames} / {N_FRAMES} "
              f"frames  ({n_bound_frames*DT_NS:.1f} ns total)")

    # Time-averaged ⟨n⟩ from the binary stack — model-free, used for cross-check.
    n_t = np.zeros(N_FRAMES, dtype=int)
    for arr in occ.values():
        n_t += arr
    mean_n = float(n_t.mean())
    print(f"\n  ⟨n_{chain}⟩ (time-averaged over 5001 frames) = {mean_n:.4f}")

    # ── Method 1 — Population SP(τ) ─────────────────────────────────────────
    section("METHOD 1 — Population autocorrelation SP(τ)")
    print("  Step 1: build origin-grid sets (here: every 1 ns = 10 frames)")
    t0_step = 10
    # Build set-per-frame from occ.
    list_of_sets = [set() for _ in range(N_FRAMES)]
    for resid, arr in occ.items():
        for fi in np.where(arr)[0]:
            list_of_sets[int(fi)].add(int(resid))

    t0_indices = list(range(0, N_FRAMES, t0_step))
    print(f"     N_origins = {len(t0_indices)}")
    print(f"     n0 at first 5 origins: "
          f"{[len(list_of_sets[t]) for t in t0_indices[:5]]}")

    print("\n  Step 2 & 3: for each (t0, τ) accumulate alive_at_τ / n0,"
          " then average across t0")
    tau_max_frames = 500   # 50 ns at dt=0.1 ns
    sums = np.zeros(tau_max_frames + 1)
    counts = np.zeros(tau_max_frames + 1)
    for t0 in t0_indices:
        n0 = len(list_of_sets[t0])
        if n0 == 0:
            continue
        alive = set(list_of_sets[t0])
        sums[0] += 1.0; counts[0] += 1
        max_tau = min(tau_max_frames, N_FRAMES - 1 - t0)
        inv_n0 = 1.0 / n0
        for tau in range(1, max_tau + 1):
            alive &= list_of_sets[t0 + tau]
            sums[tau] += len(alive) * inv_n0
            counts[tau] += 1
    sp = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    print(f"     SP(0) = {sp[0]:.4f}   (must be 1.0)")
    print(f"     SP(τ=1 ns)  = {sp[10]:.4f}")
    print(f"     SP(τ=5 ns)  = {sp[50]:.4f}")
    print(f"     SP(τ=10 ns) = {sp[100]:.4f}")
    print(f"     SP(τ=50 ns) = {sp[-1]:.4f}   (plateau ≈ c)")

    print("\n  Step 4: bi-exp fit  P(t) = α₁·exp(-t/τ₁) + α₂·exp(-t/τ₂) + c")
    from scipy.optimize import curve_fit
    tau_vec = np.arange(tau_max_frames + 1) * DT_NS

    def f2(x, u, c, t1, t2):
        a1 = u * (1 - c); a2 = (1 - u) * (1 - c)
        return a1 * np.exp(-x / t1) + a2 * np.exp(-x / t2) + c

    try:
        popt, _ = curve_fit(f2, tau_vec, sp,
                            p0=(0.5, sp[-1], 0.5, 5.0),
                            bounds=([0, 0, 1e-4, 1e-4],
                                    [1, 1, 1e3, 1e3]),
                            maxfev=20000)
        u, c, t1, t2 = popt
        a1 = u * (1 - c); a2 = (1 - u) * (1 - c)
        if t1 > t2:  # canonicalise
            t1, t2 = t2, t1; a1, a2 = a2, a1
        fitted_res = a1 * t1 + a2 * t2
        print(f"     α₁ = {a1:.4f}, τ₁ = {t1:.4f} ns      (fast)")
        print(f"     α₂ = {a2:.4f}, τ₂ = {t2:.4f} ns      (slow)")
        print(f"     c  = {c:.4f}                         (plateau)")
        print(f"     fitted residence time τ_off = α₁·τ₁ + α₂·τ₂ = {fitted_res:.4f} ns")
    except Exception as e:
        print(f"     fit failed: {e}")
        fitted_res = np.nan

    print("\n  Why this is **time-weighted**:")
    print("     A 10 ns event covers 10 origin samples; a 0.3 ns flicker covers")
    print("     ≤ 1.  → long events dominate SP(τ); the fitted τ_off answers:")
    print("     'given a randomly-timed snapshot, how long until the bound ADP leaves?'")

    # ── Method 2 — Dwell-time MLE ──────────────────────────────────────────
    section("METHOD 2 — Dwell-time MLE on individual events")
    print("  Step 1: extract events as runs of 1s in each per-molecule series")
    events = []
    for resid, arr in occ.items():
        for s, e in runs_to_events(arr, INTERMITTENCY):
            events.append({
                "resid": resid, "start_frame": s, "stop_frame": e,
                "dwell_ns": (e - s) * DT_NS,
                "left": s == 0,
                "right": e == N_FRAMES,
            })
    print(f"     N_events = {len(events)}")
    print("     First 3 events:")
    for ev in sorted(events, key=lambda e: e["start_frame"])[:3]:
        print(f"        resid {ev['resid']}: frames "
              f"{ev['start_frame']:5d}–{ev['stop_frame']:5d}  "
              f"({ev['start_frame']*DT_NS:5.1f}–"
              f"{ev['stop_frame']*DT_NS:5.1f} ns)  "
              f"dwell = {ev['dwell_ns']:.2f} ns")

    full = np.array([e["dwell_ns"] for e in events
                     if not e["left"] and not e["right"]])
    cens = np.array([e["dwell_ns"] for e in events
                     if e["right"] and not e["left"]])
    print(f"\n  Step 2: censoring split")
    print(f"     N_uncensored (event ends in trajectory) = {len(full)}")
    print(f"     N_right_censored (still bound at t=T)   = {len(cens)}")
    print(f"     Σ(uncensored dwells) = {full.sum():.2f} ns")
    print(f"     Σ(censored dwells)   = {cens.sum():.2f} ns")
    print(f"     mean(uncensored dwells) = {full.mean():.4f} ns")

    print(f"\n  Step 3: count arrivals (0→1 transitions, exclude left-censored)")
    n_arrivals = sum(1 for ev in events if not ev["left"])
    print(f"     N_arrivals = {n_arrivals}")

    print(f"\n  Step 4: MLE fit log L(π, τ₁, τ₂) =")
    print(f"          Σ_full log f(tᵢ) + Σ_cens log S(tⱼ)")
    s_fit = fit_single_exp_mle(full, cens)
    b_fit = fit_bi_exp_mle(full, cens)
    if s_fit:
        print(f"     Single-exp closed form:")
        print(f"        τ̂ = (Σ_full + Σ_cens) / N_full")
        print(f"           = ({full.sum():.2f} + {cens.sum():.2f}) / {len(full)}")
        print(f"           = {s_fit['tau']:.4f} ns ± {s_fit['perr_tau']:.4f}")
        print(f"        k_off = 1/τ = {s_fit['k_off']:.4f} ns⁻¹  "
              f"= {s_fit['k_off']*1e9:.2e} s⁻¹")
    if b_fit:
        print(f"     Bi-exp MLE (numerical):")
        print(f"        π      = {b_fit['pi']:.4f} ± {b_fit['perr_pi']:.4f}")
        print(f"        τ_fast = {b_fit['tau_fast']:.4f} ± "
              f"{b_fit['perr_tau_fast']:.4f} ns")
        print(f"        τ_slow = {b_fit['tau_slow']:.4f} ± "
              f"{b_fit['perr_tau_slow']:.4f} ns")
        print(f"        k_off_fast = {b_fit['k_off_fast']:.4f} ns⁻¹  "
              f"= {b_fit['k_off_fast']*1e9:.2e} s⁻¹")
        print(f"        k_off_slow = {b_fit['k_off_slow']:.4f} ns⁻¹  "
              f"= {b_fit['k_off_slow']*1e9:.2e} s⁻¹")

    print(f"\n  Step 5: k_on from arrival counting (Method 1 cannot do this)")
    k_on = (n_arrivals / T_TOTAL_NS) / BULK_M
    print(f"     k_on = (N_arrivals / T_total) / [bulk]")
    print(f"          = ({n_arrivals} / {T_TOTAL_NS}) / {BULK_M:.4f} M")
    print(f"          = {k_on:.4f} M⁻¹·ns⁻¹  = {k_on*1e9:.2e} M⁻¹·s⁻¹")

    if b_fit:
        print(f"\n  Step 6: K_d = k_off / k_on")
        kd_fast = b_fit['k_off_fast'] / k_on
        kd_slow = b_fit['k_off_slow'] / k_on
        print(f"     K_d (fast) = {b_fit['k_off_fast']:.4f} / {k_on:.4f} "
              f"= {kd_fast*1000:.2f} mM")
        print(f"     K_d (slow) = {b_fit['k_off_slow']:.4f} / {k_on:.4f} "
              f"= {kd_slow*1000:.4f} mM   ← physically meaningful affinity")

    print("\n  Why this is **event-weighted**:")
    print("     Every event contributes one likelihood term.  A 0.3 ns flicker")
    print("     and a 10 ns binder count equally.  → MLE τ̂ values are TRUE")
    print("     per-event rate constants, suitable for k_on/k_off/K_d in M⁻¹·s⁻¹.")

    # ── Cross-validation ───────────────────────────────────────────────────
    section("CROSS-VALIDATION  ⟨n⟩ from each method")
    print("  Both methods must agree on the integrated quantity:")
    print(f"     direct time-average:                {mean_n:.4f}")
    sigma_dwells = full.sum() + cens.sum()
    n_check = sigma_dwells / T_TOTAL_NS
    print(f"     Σ(dwells)/T_total ="
          f" ({full.sum():.2f}+{cens.sum():.2f})/{T_TOTAL_NS} = "
          f"{n_check:.4f}")
    print(f"     T1 per_chain_summary ⟨n⟩:           "
          f"(see results_v3/T1_per_chain_pocket_{buf}/per_chain_summary.csv)")

    section("Summary")
    print(f"  Method 1 (population): τ_off (fitted) ≈ {fitted_res:.3f} ns "
          f"— time-weighted")
    if b_fit:
        em = b_fit['pi'] * b_fit['tau_fast'] + (1 - b_fit['pi']) * b_fit['tau_slow']
        print(f"  Method 2 (dwell MLE): ⟨τ⟩_event ≈ {em:.3f} ns "
              f"— event-weighted")
        print(f"                         τ_slow ≈ {b_fit['tau_slow']:.3f} ns "
              f"(real binders)")
        print(f"                         k_on  = {k_on*1e9:.2e} M⁻¹s⁻¹")
        print(f"                         k_off (slow) = "
              f"{b_fit['k_off_slow']*1e9:.2e} s⁻¹")
        print(f"                         K_d (slow)   = {kd_slow*1000:.2f} mM")
    print()


if __name__ == "__main__":
    main()
