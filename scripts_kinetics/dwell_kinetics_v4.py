#!/usr/bin/env python3
"""Method 2 dwell-time kinetics for the v4 per-region datasets.

For every results_v4/v4_<dataset>/ run, iterate over the per-region
contact files in _intermediate/contacts/<region>.npz, reconstruct
per-molecule binary occupancy, extract dwell intervals (right-censored),
fit single-exp+c and bi-exp+c survival models by maximum likelihood,
and write outputs to results_v4/v4_<dataset>_method2/ — parallel to the
Method 1 directory so the names are clearly differentiated.

CSV columns mirror Method 1 where a 1:1 mapping exists. The plot style
matches Method 1 (fit_<region>.svg + fit_<region>_equations.svg, same
colors, same hollow-bar overlay, same dashed-darkred / dotted-darkorange
fit curves).
"""

from __future__ import annotations

import csv
import gc
import glob
import os
import re
import sys
import time
import traceback
from multiprocessing import Pool, get_context

import numpy as np


def _read_existing_csv(path):
    """Read CSV into list of dict (preserving str values). Empty list if missing."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            # Promote numeric strings back to floats where possible (only
            # so write_csv can re-format them through fmt_val).
            d = {}
            for k, v in row.items():
                if v == "":
                    d[k] = ""
                else:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
            out.append(d)
    return out

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dwell_time_kinetics import (  # noqa: E402
    fit_single_exp_c_mle,
    fit_bi_exp_c_mle,
    fmt_val,
    INTERMITTENCY,
)

# Shared event/timebase primitives — see kinetics/events.py for the
# canonical implementations; these re-exports keep the v4 driver readable.
from kinetics.events import (  # noqa: E402
    infer_sample_dt_ns as _infer_sample_dt_ns,
    contact_timebase,
    load_contact_matrix as _load_contact_matrix,
    contact_runs as _contact_runs,
    load_occupancy,
    collect_events_region as _shared_collect_events_region,
    avg_count_per_frame,
    empirical_survival as empirical_survival_fast,
    empirical_survival_grouped,
    r2_against_KM,
    r2_against_KM_grouped,
)
empirical_survival = empirical_survival_fast


def _mobile_rt(alpha_fast, tau_fast, alpha_slow, tau_slow):
    """Conditional mean residence time of the *mobile* sub-population.

    ⟨t⟩|mobile = (α_f τ_f + α_s τ_s) / (α_f + α_s) = fitted_res_time / (1−c).
    Used in both MLE and KM-LS bi-exp result composers. nan if α_f+α_s ≤ 0.
    """
    try:
        a_sum = float(alpha_fast) + float(alpha_slow)
    except (TypeError, ValueError):
        return float("nan")
    if not (a_sum > 0):
        return float("nan")
    return (float(alpha_fast) * float(tau_fast)
            + float(alpha_slow) * float(tau_slow)) / a_sum


# Method-1 LS fitters for the KM-LS path. Method 2 KM-LS fits the grouped
# Kaplan-Meier survival curve with the same single+c / bi+c models, so
# Method 2 outputs include a direct visual analogue to Method 1's SP fit.
import importlib.util as _ilu  # noqa: E402
_fitting_spec = _ilu.spec_from_file_location(
    "kinetics_fitting", os.path.join(HERE, "fitting.py"))
_fitting = _ilu.module_from_spec(_fitting_spec)
_fitting_spec.loader.exec_module(_fitting)
fit_single_exp_ls = _fitting.fit_single_exp
fit_bi_exp_ls = _fitting.fit_bi_exp
f1_constrained = _fitting.f1_constrained
f2_constrained = _fitting.f2_constrained


DEFAULT_TRAJ_DT_NS = 0.01

# Conversion: [L] (M) = N × 1e24 / (N_A × V[nm³]) = N × N_M_PER_NM3 / V[nm³]
N_M_PER_NM3 = 1.0e24 / 6.02214076e23  # ≈ 1.66054 (M·nm³ per molecule)


# Per-dataset bulk-binding denominators. Values taken from each system's em.gro:
#   v_box_nm3 = product of the cubic box edge from the last line of em.gro
#   n_solute  = unique residue count for the solute resname in em.gro
# Used to compute k_on_bulk = N_arr / ([L_bulk] × T_traj) per region.
# The "single_site, V_box, T_traj" assumption is recorded in k_on_bulk_assumes.
DATASET_METADATA = {
    "v4_adp_0216":   {"solute": "ADP", "n_solute":    300, "v_box_nm3": 7019.0, "buffer": "EDDA"},
    "v4_adp_0311":   {"solute": "ADP", "n_solute":    300, "v_box_nm3": 7019.0, "buffer": "AMAC"},
    "v4_ace_0216":   {"solute": "ACE", "n_solute":   1700, "v_box_nm3": 7019.0, "buffer": "EDDA"},
    "v4_ace_0311":   {"solute": "ACE", "n_solute":    850, "v_box_nm3": 7019.0, "buffer": "AMAC"},
    "v4_dda_0216":   {"solute": "DDA", "n_solute":    425, "v_box_nm3": 7019.0, "buffer": "EDDA"},
    "v4_mda_0216":   {"solute": "MDA", "n_solute":    425, "v_box_nm3": 7019.0, "buffer": "EDDA"},
    "v4_nh4_0311":   {"solute": "NH4", "n_solute":    851, "v_box_nm3": 7019.0, "buffer": "AMAC"},
    "v4_water_0216": {"solute": "SOL", "n_solute": 100000, "v_box_nm3": 7019.0, "buffer": "EDDA"},
    "v4_water_0311": {"solute": "SOL", "n_solute": 100000, "v_box_nm3": 7019.0, "buffer": "AMAC"},
}


def conc_bulk_M(n_solute, v_box_nm3):
    """Bulk solute concentration in molarity (mol/L)."""
    return n_solute * N_M_PER_NM3 / v_box_nm3


def load_traj_dt_ns(dataset_dir, default=DEFAULT_TRAJ_DT_NS):
    """Parse dt(traj) from run_settings.log, falling back to 10 ps frames."""
    log = os.path.join(dataset_dir, "run_settings.log")
    if not os.path.exists(log):
        return float(default)
    pat = re.compile(r"dt\(traj\)\s*=\s*([-\d.eE+]+)\s+ns/frame")
    with open(log) as f:
        for line in f:
            m = pat.search(line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return float(default)
    return float(default)


# Preload the shared plotting module at top-level (NOT lazily inside
# plot_fit) so matplotlib's backend + metadata init happens once in the
# main process before any worker forks. This is the V5-W fix for the
# SIGSEGV-on-first-plot-import that triggered the 2026-05-10 hang.
try:
    import plotting as _plotting  # noqa: E402
    plot_survival_fit = _plotting.plot_survival_fit
except ImportError:
    from . import plotting as _plotting
    plot_survival_fit = _plotting.plot_survival_fit

from kinetics.datasets import (  # noqa: E402
    V4_ROOT, M2_OUT_SUFFIX,
    discover_v4_datasets,
    method2_output_dir,
    contacts_dir,
    list_contact_regions,
)
# Backwards-compat alias for the few external callers that still reference
# the old name.
OUT_SUFFIX = M2_OUT_SUFFIX

# Cap the number of events fed to the MLE optimizer. Bi-exp parameter precision
# scales as 1/sqrt(N), so above ~200k samples the fit gains nothing while the
# numerical_hessian + L-BFGS-B inner loops allocate ~80 MB temps per nll call.
# Without this cap, water/protein_shell (~10M events) hit ~2 GB peak transient
# and OOM-killed the process. Setting None disables subsampling.
MAX_EVENTS_FOR_FIT = 200_000


def _maybe_subsample(t_full, t_cens, cap=MAX_EVENTS_FOR_FIT, seed=0):
    """Random subsample if total event count exceeds `cap`. Stratified by
    type (full vs censored) so the censoring fraction is preserved."""
    if cap is None:
        return t_full, t_cens
    n = len(t_full) + len(t_cens)
    if n <= cap:
        return t_full, t_cens
    rng = np.random.default_rng(seed)
    keep_frac = cap / n
    sel_full = rng.random(len(t_full)) < keep_frac
    sel_cens = rng.random(len(t_cens)) < keep_frac
    return t_full[sel_full], t_cens[sel_cens]


# Per-region event extraction is shared with kinetics/events.py.
def collect_events_region(npz_path, traj_dt_ns=DEFAULT_TRAJ_DT_NS):
    """Forward to kinetics.events.collect_events_region with v4 INTERMITTENCY."""
    return _shared_collect_events_region(
        npz_path, traj_dt_ns=traj_dt_ns, intermittency=INTERMITTENCY)


# ── Events CSV cache (V5-7c) ────────────────────────────────────────────
# Per-region dwell events written as text so replots at any x_max / bin
# width need only this file — no contact npz reload, no contact_runs pass.
# Two columns: t_ns (float, ns), censored (0=completed, 1=right-censored).
# Supports plain .csv and gzipped .csv.gz transparently.
def events_cache_path(npz_path, fmt="csv.gz"):
    """events npz <dataset>/_intermediate/contacts/<region>.npz
    →   <dataset>/_intermediate/events_cache/<region>.<fmt>
    """
    import os as _os
    contacts_dir = _os.path.dirname(npz_path)
    inter_dir    = _os.path.dirname(contacts_dir)
    region       = _os.path.splitext(_os.path.basename(npz_path))[0]
    return _os.path.join(inter_dir, "events_cache", f"{region}.{fmt}")


def write_events_csv(t_full, t_cens, path):
    """Write per-event CSV (or CSV.gz). t_full=completed dwells, t_cens=censored."""
    import os as _os, gzip as _gzip
    import numpy as _np
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    n_f, n_c = len(t_full), len(t_cens)
    data = _np.column_stack([
        _np.concatenate([_np.asarray(t_full, dtype=_np.float64),
                         _np.asarray(t_cens, dtype=_np.float64)]) if (n_f or n_c) else _np.zeros((0,), dtype=_np.float64),
        _np.concatenate([_np.zeros(n_f, dtype=_np.int8),
                         _np.ones(n_c,  dtype=_np.int8)])           if (n_f or n_c) else _np.zeros((0,), dtype=_np.int8),
    ]) if (n_f or n_c) else _np.zeros((0, 2))
    opener = (lambda p: _gzip.open(p, "wt", compresslevel=6)) if path.endswith(".gz") else (lambda p: open(p, "w"))
    with opener(path) as f:
        f.write("t_ns,censored\n")
        if len(data):
            _np.savetxt(f, data, fmt=["%.6f", "%d"], delimiter=",")


def read_events_csv(path):
    """Inverse of write_events_csv. Returns (full, cens) as float64 arrays."""
    import numpy as _np
    arr = _np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if arr.size == 0:
        return _np.zeros((0,), dtype=_np.float64), _np.zeros((0,), dtype=_np.float64)
    full = arr[arr[:, 1] == 0, 0].astype(_np.float64)
    cens = arr[arr[:, 1] == 1, 0].astype(_np.float64)
    return full, cens


def load_events(npz_path, traj_dt_ns=DEFAULT_TRAJ_DT_NS, prefer_cache=True):
    """Get (full, cens) for a region. Prefer cached CSV/CSV.gz when available.

    Falls back to collect_events_region (Phase A re-derivation from the
    contact npz) only when no cache file is present. Cache lookup tries
    .csv.gz first, then .csv.
    """
    if prefer_cache:
        for fmt in ("csv.gz", "csv"):
            p = events_cache_path(npz_path, fmt=fmt)
            if os.path.exists(p):
                full, cens = read_events_csv(p)
                return full, cens
    full, cens, _, _ = collect_events_region(npz_path, traj_dt_ns=traj_dt_ns)
    return full, cens


def per_molecule_rates(npz_path, traj_dt_ns=DEFAULT_TRAJ_DT_NS,
                       cached_runs=None):
    """Compute per-molecule on/off hazards directly from the binary contact array.

    ``cached_runs`` lets the caller avoid redoing ``contact_runs`` (which
    is the dominant cost on water ``protein_shell``: ~150 s for 1.5 B
    records). Pass the tuple returned by ``contact_runs(npz)``.

    Returns dict with:
      lambda_on_visiting = (total arrivals) / (total free time across visiting molecules)
      lambda_off         = (total completed dissociations) / (total bound time)
      on_rate_scope      = "visiting_molecules" (denominator scope tag)
      n_arr_total, n_diss_total: total transition counts
      T_free_total_ns, T_bound_total_ns: cumulative times
      n_molecules_visiting: how many distinct solutes ever contacted this cavity

    Scope: λ_on_visiting uses only molecules that visited the cavity at least
    once (those represented in the npz) as the denominator. This is a
    per-visiting-molecule arrival hazard, NOT a macroscopic second-order k_on.
    A true bulk-normalized k_on would require [L] (bulk concentration), the
    full candidate-molecule denominator, site count, and total unbound site
    time — none of which this function computes.
    """
    if cached_runs is not None:
        starts, stops, n_bound_samples, dt_ns, n_samples, n_unique = cached_runs
    else:
        starts, stops, n_bound_samples, dt_ns, n_samples, n_unique = _contact_runs(
            npz_path, traj_dt_ns=traj_dt_ns
        )
    if starts.size == 0:
        return {"lambda_on_visiting": float("nan"), "lambda_off": float("nan"),
                "on_rate_scope": "visiting_molecules",
                "n_arr_total": 0, "n_diss_total": 0,
                "T_free_total_ns": 0.0, "T_bound_total_ns": 0.0,
                "n_molecules_visiting": 0,
                "dt_ns": float(dt_ns),
                "n_contact_frames": int(n_samples)}

    # Arrivals: count of +1 edges (excluding "left-censored" — molecules already
    # bound at frame 0). starts == 0 means start at trajectory boundary, not a
    # true arrival from free state.
    arrivals_mask = starts > 0
    n_arr_total = int(np.count_nonzero(arrivals_mask))

    # Completed dissociations: -1 edges where stop < n_samples (not still bound at end)
    diss_mask = stops < n_samples
    n_diss_total = int(np.count_nonzero(diss_mask))

    # Free / bound time across all visiting molecules
    bound_frames = int(n_bound_samples)
    free_frames = int(n_unique * n_samples - bound_frames)  # visiting molecules only
    T_bound_total = bound_frames * dt_ns
    T_free_total = free_frames * dt_ns

    lambda_on_visiting = n_arr_total / T_free_total if T_free_total > 0 else float("nan")
    lambda_off = n_diss_total / T_bound_total if T_bound_total > 0 else float("nan")

    return {"lambda_on_visiting": float(lambda_on_visiting),
            "lambda_off": float(lambda_off),
            "on_rate_scope": "visiting_molecules",
            "n_arr_total": n_arr_total,
            "n_diss_total": n_diss_total,
            "T_free_total_ns": float(T_free_total),
            "T_bound_total_ns": float(T_bound_total),
            "n_molecules_visiting": int(n_unique),
            "dt_ns": float(dt_ns),
            "n_contact_frames": int(n_samples)}


# ── KM-LS path: M1-style LS fit on grouped KM survival ────────────────────
KM_LS_MAX_POINTS = 400   # cap before LS fit; protein_shell can have 20k+ pts


def _downsample_km_log(t, S, n_target):
    """Log-spaced downsample of a KM curve. Always keeps t=0 and the last
    point (so the plateau is preserved)."""
    if len(t) <= n_target:
        return t, S
    # Reserve t=0 anchor and final point.
    t_in = np.asarray(t, dtype=float)
    S_in = np.asarray(S, dtype=float)
    inner_t = t_in[1:-1]
    inner_S = S_in[1:-1]
    if inner_t.size <= n_target - 2:
        return t_in, S_in
    # Log-spaced indices over the inner range.
    log_t = np.log(np.maximum(inner_t, 1e-12))
    target_log = np.linspace(log_t[0], log_t[-1], n_target - 2)
    idx = np.searchsorted(log_t, target_log)
    idx = np.clip(idx, 0, inner_t.size - 1)
    idx = np.unique(idx)
    t_out = np.concatenate([[t_in[0]], inner_t[idx], [t_in[-1]]])
    S_out = np.concatenate([[S_in[0]], inner_S[idx], [S_in[-1]]])
    return t_out, S_out


KM_INPUT_MAX_EVENTS = 1_000_000   # cap events fed to KM grouping

# V5-4: minimum n_full to attempt each fit. Below the single threshold,
# the row is short-circuited to insufficient_data with no fitted metrics.
MIN_FULL_EVENTS_SINGLE = 5    # below this → insufficient_data
MIN_FULL_EVENTS_MLE_BI = 5    # MLE bi+c requires ≥5 (4 params + 1)
MIN_FULL_EVENTS_LS_BI = 4     # LS bi+c needs ≥4 unique grouped KM points


def fit_km_ls(full, cens, max_points=KM_LS_MAX_POINTS,
              input_cap=KM_INPUT_MAX_EVENTS):
    """Fit single+c and bi+c models to grouped KM survival via least squares.

    Two scaling caps to keep water `protein_shell` (≥10⁸ events) tractable:

    1. ``input_cap`` — if total events exceed it, randomly subsample
       (stratified by full vs censored) before grouping. With ≥10⁵ events
       the grouped KM curve is statistically indistinguishable; we don't
       lose anything biologically meaningful.

    2. ``max_points`` — log-downsample the grouped KM curve before LS so
       SciPy ``curve_fit`` × multi-start × maxfev stays bounded.

    Returns (s_fit, b_fit, t_km_full, S_km_full) where the *full* grouped
    KM (post-subsampling, pre-log-downsample) is returned for plotting.
    """
    full_use, cens_use = full, cens
    n_total = len(full) + len(cens)
    if n_total > input_cap:
        rng = np.random.default_rng(0)
        keep = input_cap / n_total
        sf = rng.random(len(full)) < keep
        sc = rng.random(len(cens)) < keep
        full_use, cens_use = full[sf], cens[sc]
    t_km, S_km = empirical_survival_grouped(full_use, cens_use)
    if len(t_km) >= 2:
        t_fit, S_fit = _downsample_km_log(t_km, S_km, max_points)
    else:
        t_fit, S_fit = t_km, S_km
    s_fit = fit_single_exp_ls(t_fit, S_fit) if len(t_fit) >= 2 else None
    b_fit = fit_bi_exp_ls(t_fit, S_fit) if len(t_fit) >= 4 else None
    return s_fit, b_fit, t_km, S_km


def select_model_bic(s_bic, b_bic, threshold=2.0):
    """Pick `biexp_c` only if BIC_single - BIC_bi > threshold; else single_c
    or inconclusive. Returns (model_chosen, selection_status, delta_BIC).

    selection_status is now a category only (no embedded number) — the V4
    plan asked for `delta_BIC` to live in its own column.
    """
    if s_bic is None and b_bic is None:
        return "none", "no_fit", float("nan")
    if b_bic is None:
        return "single_c", "single_only", float("nan")
    if s_bic is None:
        return "biexp_c", "bi_only", float("nan")
    delta = float(s_bic) - float(b_bic)
    if delta > threshold:
        return "biexp_c", "bi_preferred", delta
    if delta < -threshold:
        return "single_c", "single_preferred", delta
    return "single_c", "inconclusive", delta


def biological_resolved(alpha1, alpha2, tau1, tau2, delta_BIC,
                        bic_threshold=2.0, tau_ratio_min=3.0,
                        minor_min=0.05):
    """Legacy V4 single-flag bi+c resolution check. Kept as alias.

    V5 splits this into separate columns; see ``biological_flags`` for
    the publication-grade flag set.
    """
    try:
        a1 = float(alpha1); a2 = float(alpha2)
        t1 = float(tau1); t2 = float(tau2)
    except (TypeError, ValueError):
        return float("nan"), float("nan"), False
    if not (np.isfinite(a1) and np.isfinite(a2)
            and np.isfinite(t1) and np.isfinite(t2)
            and t1 > 0 and t2 > 0):
        return float("nan"), float("nan"), False
    tau_ratio = max(t1, t2) / min(t1, t2)
    minor = min(a1, a2)
    bic_ok = (np.isfinite(delta_BIC) and float(delta_BIC) > bic_threshold)
    resolved = (bic_ok and tau_ratio >= tau_ratio_min
                and minor >= minor_min)
    return float(tau_ratio), float(minor), bool(resolved)


def biological_flags(alpha1, alpha2, tau1, tau2, delta_BIC,
                     full_dwells=None, r2_km_grouped=float("nan"),
                     fit_kind="MLE",
                     bic_threshold=2.0, tau_ratio_min=3.0,
                     minor_min=0.05, tail_min=3,
                     r2_threshold_mle=0.90, r2_threshold_ls=0.98):
    """Publication-grade resolution flags for a bi+c fit (V5 P0-#4).

    Returns dict with the columns:
      - statistically_selected: ΔBIC > 2
      - component_separated:    tau_ratio ≥ 3
      - minor_component_supported: minor amp ≥ 0.05
      - tail_supported:         ≥ tail_min completed dwells longer than
                                max(3·τ_fast, 1 ns)
      - curve_consistent:       R²_KM_grouped ≥ 0.90 (MLE) or 0.98 (KM-LS)
      - reportable_two_state:   conjunction of all five
      - tau_ratio, minor_component_fraction, n_full_tail (numeric)

    fit_kind is "MLE" or "KM-LS" — sets the R² threshold.
    full_dwells is the array of completed dwell times (after subsampling
    that the fitter saw). Pass None to skip tail check.
    """
    try:
        a1 = float(alpha1); a2 = float(alpha2)
        t1 = float(tau1); t2 = float(tau2)
        valid = (np.isfinite(a1) and np.isfinite(a2)
                 and np.isfinite(t1) and np.isfinite(t2)
                 and t1 > 0 and t2 > 0)
    except (TypeError, ValueError):
        valid = False; a1=a2=t1=t2=float("nan")
    if not valid:
        return {
            "tau_ratio": float("nan"),
            "minor_component_fraction": float("nan"),
            "n_full_tail": 0,
            "statistically_selected": False,
            "component_separated": False,
            "minor_component_supported": False,
            "tail_supported": False,
            "curve_consistent": False,
            "reportable_two_state": False,
        }
    tau_ratio = max(t1, t2) / min(t1, t2)
    minor = min(a1, a2)
    statistically_selected = (np.isfinite(delta_BIC)
                              and float(delta_BIC) > bic_threshold)
    component_separated = tau_ratio >= tau_ratio_min
    minor_component_supported = minor >= minor_min
    # Tail support: count completed dwells beyond the "fast collapse"
    # window. If no dwells passed in (e.g. KM-LS row that doesn't
    # propagate full[]), conservatively mark False.
    tau_fast = min(t1, t2)
    tail_threshold = max(3.0 * tau_fast, 1.0)
    if full_dwells is not None and len(full_dwells) > 0:
        n_tail = int(np.count_nonzero(np.asarray(full_dwells) > tail_threshold))
    else:
        n_tail = 0
    tail_supported = n_tail >= tail_min
    r2_thr = r2_threshold_mle if fit_kind == "MLE" else r2_threshold_ls
    curve_consistent = (np.isfinite(r2_km_grouped)
                        and float(r2_km_grouped) >= r2_thr)
    reportable = (statistically_selected and component_separated
                  and minor_component_supported and tail_supported
                  and curve_consistent)
    return {
        "tau_ratio": float(tau_ratio),
        "minor_component_fraction": float(minor),
        "n_full_tail": int(n_tail),
        "statistically_selected": bool(statistically_selected),
        "component_separated": bool(component_separated),
        "minor_component_supported": bool(minor_component_supported),
        "tail_supported": bool(tail_supported),
        "curve_consistent": bool(curve_consistent),
        "reportable_two_state": bool(reportable),
    }


# ── Method 2 plot — delegates to the shared shared core in plotting.py
def plot_fit(region, full, cens, s_fit, b_fit, plot_dir,
             method_label="MLE",
             dpi=150, alpha=0.65, n_bins=None, bin_spacing_factor=0.7,
             base_fontsize=16, hard_cap=50.0, separate_equations=False,
             bin_width_ns=None,
             rates=None,
             x_max=None):
    """Method 2 plot — observed grouped KM bars + bi+c overlay.

    Thin shim over `plotting.plot_survival_fit` so M1 SP-LS, M2 MLE,
    and M2 KM-LS all use the same renderer with the same tick-aware
    bin width and merged-equation legend by default.

    `n_bins` is accepted for back-compat but ignored — bins are now
    derived from tick-aware `bin_width_ns = nice_tick_interval(x_max)/5`.

    `rates` is the per-region rate dict (from `per_molecule_rates`)
    plus optional bulk fields; when provided, the bi+c equation legend
    gains k_off_fast, k_off_slow, λ_on,visiting, and k_on,bulk lines.
    """
    if len(full) == 0 or b_fit is None:
        return False
    # Build the observed grouped KM curve once here so both the plot and
    # the in-CSV R²_KM_grouped use the *same* curve.
    t_obs, S_obs = empirical_survival_grouped(full, cens)
    # plotting is preloaded at module import (top of file) so we don't
    # take the matplotlib-import path inside a worker process.
    return plot_survival_fit(
        t_obs=t_obs, S_obs=S_obs,
        s_fit=s_fit, b_fit=b_fit,
        name=region, outdir=plot_dir,
        method_label=method_label,
        full_dwells=full,
        bin_width_ns=bin_width_ns,
        x_max=x_max, hard_cap=hard_cap,
        base_fontsize=base_fontsize,
        alpha=alpha, dpi=dpi,
        separate_equations=separate_equations,
        rates=rates,
    )

# ── Per-region summary text (Method-1-style header + Method-2 fit body) ────
def write_region_summary(region, desc, full, cens, n_arr, n_events,
                          avg_n, std_n, s_fit, b_fit, out_path,
                          dt_ns, n_contact_frames,
                          s_ls=None, b_ls=None,
                          mle_chosen="", mle_status="",
                          ls_chosen="", ls_status="",
                          km_unique_points=0):
    lines = []
    lines.append(f"[{region}] contact samples 0:{n_contact_frames}")
    if desc:
        lines.append(f"Description: {desc}")
    lines.append("")
    lines.append("Method: dwell-time MLE (single+c and bi+c, right-censored)")
    lines.append(f"  dt(analysis) = {dt_ns:.4f} ns/contact-sample")
    lines.append(f"  Intermittency = {INTERMITTENCY} frames")
    lines.append("")
    lines.append("Event counts:")
    lines.append(f"  N events (total)     = {n_events}")
    lines.append(f"  N events (full, used)= {len(full)}")
    lines.append(f"  N events (right-cens)= {len(cens)}")
    lines.append(f"  N arrivals           = {n_arr}")
    lines.append(f"  Average # bound molecules per frame: "
                 f"{avg_n:.3f} +/- {std_n:.3f}")
    lines.append("")
    if len(full) > 0:
        lines.append(f"Mean dwell (uncens)  = {float(np.mean(full)):.4f} ns")
        lines.append(f"Median dwell         = {float(np.median(full)):.4f} ns")
        lines.append("")
    if s_fit is not None:
        lines.append("Single-exp+c MLE  S(t) = a*exp(-t/tau) + c:")
        lines.append(f"  alpha = {s_fit['alpha']:.4f} +/- {s_fit.get('perr_alpha', float('nan')):.4f}")
        lines.append(f"  c     = {s_fit['c']:.4f} +/- {s_fit.get('perr_c', float('nan')):.4f}")
        lines.append(f"  tau   = {s_fit['tau']:.4f} ns +/- {s_fit['perr_tau']:.4f}")
        lines.append(f"  k_off = 1/tau = {s_fit['k_off']:.4f} ns^-1 "
                     f"= {s_fit['k_off']*1e9:.3e} s^-1")
        lines.append(f"  R^2 vs KM = {s_fit.get('r2_km', float('nan')):.4f}")
        lines.append(f"  AIC = {s_fit['AIC']:.2f}, AICc = {s_fit['AICc']:.2f}, "
                     f"BIC = {s_fit['BIC']:.2f}")
        lines.append("")
    if b_fit is not None:
        lines.append("Bi-exp+c MLE:  S(t) = a_f*exp(-t/tau_f) + a_s*exp(-t/tau_s) + c:")
        lines.append(f"  a_fast = {b_fit['alpha_fast']:.4f} +/- {b_fit['perr_alpha_fast']:.4f}")
        lines.append(f"  a_slow = {b_fit['alpha_slow']:.4f} +/- {b_fit['perr_alpha_slow']:.4f}")
        lines.append(f"  c      = {b_fit['c']:.4f} +/- {b_fit.get('perr_c', float('nan')):.4f}")
        lines.append(f"  tau_fast= {b_fit['tau_fast']:.4f} ns "
                     f"+/- {b_fit['perr_tau_fast']:.4f}")
        lines.append(f"  tau_slow= {b_fit['tau_slow']:.4f} ns "
                     f"+/- {b_fit['perr_tau_slow']:.4f}")
        lines.append(f"  k_off_fast = {b_fit['k_off_fast']:.4f} ns^-1 "
                     f"= {b_fit['k_off_fast']*1e9:.3e} s^-1")
        lines.append(f"  k_off_slow = {b_fit['k_off_slow']:.4f} ns^-1 "
                     f"= {b_fit['k_off_slow']*1e9:.3e} s^-1")
        lines.append(f"  <tau> mix = {b_fit['mean_dwell_ns']:.4f} ns "
                     f"(fitted residence time)")
        lines.append(f"  t1/2 fast = {b_fit['t_half_fast']:.4f} ns, "
                     f"t1/2 slow = {b_fit['t_half_slow']:.4f} ns")
        lines.append(f"  R^2 vs KM = {b_fit.get('r2_km', float('nan')):.4f}")
        lines.append(f"  AIC = {b_fit['AIC']:.2f}, AICc = {b_fit['AICc']:.2f}, "
                     f"BIC = {b_fit['BIC']:.2f}")
        lines.append("")
    if s_fit is not None and b_fit is not None:
        d_aic = s_fit["AIC"] - b_fit["AIC"]
        d_bic = s_fit["BIC"] - b_fit["BIC"]
        verdict_aic = ("bi-exp preferred" if d_aic > 2
                       else "inconclusive" if d_aic > -2
                       else "single preferred")
        verdict_bic = ("bi-exp preferred" if d_bic > 2
                       else "inconclusive" if d_bic > -2
                       else "single preferred")
        lines.append("Model selection (MLE single vs bi-exp):")
        lines.append(f"  Delta AIC = {d_aic:.2f}  ({verdict_aic})")
        lines.append(f"  Delta BIC = {d_bic:.2f}  ({verdict_bic})")
        lines.append(f"  model_chosen = {mle_chosen}  ({mle_status})")
        lines.append("")
    if s_ls is not None or b_ls is not None:
        lines.append("KM-LS fit (M1-style LS on grouped KM survival):")
        lines.append(f"  KM unique points: {km_unique_points}")
        if s_ls is not None:
            lines.append(f"  single+c: τ={s_ls['tau']:.3f}±{s_ls['perr_tau']:.3f} ns,"
                         f" c={s_ls['c']:.4f},  R²={s_ls.get('r_squared', float('nan')):.4f},"
                         f" BIC={s_ls.get('bic', float('nan')):.2f}")
        if b_ls is not None:
            lines.append(f"  bi+c:    τ_f={b_ls['tau1']:.3f}, τ_s={b_ls['tau2']:.3f},"
                         f" u={b_ls['u']:.3f}, c={b_ls['c']:.4f},"
                         f"  R²={b_ls.get('r_squared', float('nan')):.4f},"
                         f" BIC={b_ls.get('bic', float('nan')):.2f}")
        lines.append(f"  model_chosen = {ls_chosen}  ({ls_status})")
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── CSV row builders (mirror Method 1 columns) ─────────────────────────────
# Method 2 emits four CSVs per dataset:
#   method2_mle_single_exp.csv   — dwell-MLE, single+c
#   method2_mle_bi_exp.csv       — dwell-MLE, bi+c
#   method2_km_ls_single_exp.csv — LS on grouped KM survival, single+c
#   method2_km_ls_bi_exp.csv     — LS on grouped KM survival, bi+c
# The legacy single_exp_fitting_results.csv / bi_exp_fitting_results.csv
# files are kept as compatibility aliases of the MLE outputs.
SINGLE_MLE_COLS = [
    "Region", "avg_count", "std_count",
    "n_full", "n_cens", "n_arrivals",
    "fit_status", "model_chosen", "selection_status", "delta_BIC",
    "alpha", "c",
    "tau", "perr_tau", "k_off",
    "mean_dwell_ns", "median_dwell_ns",
    "mobile_res_time",
    "R2_KM_eventpoints", "R2_KM_grouped",
    "AIC", "AICc", "BIC", "nll", "optimizer_status",
    "fit_event_count", "full_fit_count", "cens_fit_count", "subsample_seed",
]
# Legacy alias (used by replot/build_v4_combined_summary readers).
SINGLE_COLS = SINGLE_MLE_COLS
BIEXP_MLE_COLS = [
    "Region", "avg_count", "std_count",
    "n_full", "n_cens", "n_arrivals",
    "fit_status", "model_chosen", "selection_status", "delta_BIC",
    "alpha1", "u", "tau1", "perr_tau1",
    "alpha2", "tau2", "perr_tau2",
    "c", "perr_c",
    "event_alpha1", "event_alpha2", "event_c",
    "pop_alpha1_finite", "pop_alpha2_finite",
    "k_off_fast", "k_off_slow", "k_off_mean",
    "fitted_res_time", "mobile_res_time", "apparent_res_time",
    "t_half_fast", "t_half_slow",
    # Biological resolution flags (V4 single + V5 split — see V5 plan §P0-#4).
    "tau_ratio", "minor_component_fraction", "biologically_resolved",
    "n_full_tail",
    "statistically_selected", "component_separated",
    "minor_component_supported", "tail_supported",
    "curve_consistent", "reportable_two_state",
    "R2_KM_eventpoints", "R2_KM_grouped",
    "AIC", "AICc", "BIC", "nll", "nll_pure_bi", "optimizer_status",
    # Per-molecule kinetic hazards derived directly from contacts[t,m].
    # All in ns^-1 (multiply by 1e9 for s^-1).
    # NOTE: lambda_on_visiting is per visiting-molecule arrival hazard,
    # NOT a macroscopic second-order k_on. See on_rate_scope column.
    "lambda_on_visiting", "lambda_off", "on_rate_scope",
    "lambda_on_visiting_fast", "lambda_on_visiting_slow", "lambda_on_visiting_c",
    "k_off_population",            # finite-population off flux per bound molecule
    "k_off_initial_hazard",         # event-age-zero hazard = sum event_alpha_i*k_i
    "K_eq_visiting_fast", "K_eq_visiting_slow", "K_eq_visiting_total_finite",
    "bound_fraction_observed", "bound_free_ratio_observed",
    "bound_free_ratio_model_finite",
    "n_arr_total", "n_diss_total",
    "T_free_total_ns", "T_bound_total_ns",
    "n_molecules_visiting", "dt_ns", "n_contact_frames",
    # Macroscopic second-order on-rate using bulk-solute denominator.
    #   k_on_bulk [M^-1 s^-1] = N_arrivals / ([L_bulk] × T_traj)
    # treats each cavity as a single site under pseudo-first-order conditions.
    # k_on_bulk_assumes records the assumptions; bulk_solute / n_solute_total /
    # v_box_nm3 / conc_bulk_M / T_traj_ns are the inputs to the formula.
    "bulk_solute", "n_solute_total", "v_box_nm3", "conc_bulk_M",
    "T_traj_ns", "k_on_bulk_M_per_s", "k_on_bulk_assumes",
    "fit_event_count", "full_fit_count", "cens_fit_count", "subsample_seed",
]
BIEXP_COLS = BIEXP_MLE_COLS

# KM-LS columns: same single+c / bi+c models, but fit by least-squares to
# the grouped Kaplan-Meier survival curve. No event-level rates; the LS
# objective is RSS against grouped KM points.
SINGLE_KM_LS_COLS = [
    "Region", "avg_count", "std_count",
    "n_full", "n_cens", "n_arrivals",
    "fit_status", "model_chosen", "selection_status", "delta_BIC",
    "alpha", "tau", "perr_tau", "c", "perr_c",
    "apparent_res_time", "mobile_res_time",
    "R2_KM_grouped", "RSS", "AIC", "AICc", "BIC",
    "km_unique_points",
]
BIEXP_KM_LS_COLS = [
    "Region", "avg_count", "std_count",
    "n_full", "n_cens", "n_arrivals",
    "fit_status", "model_chosen", "selection_status", "delta_BIC",
    "alpha1", "u", "tau1", "perr_tau1",
    "alpha2", "tau2", "perr_tau2",
    "c", "perr_c",
    "fitted_res_time", "mobile_res_time", "apparent_res_time",
    "t_half_fast", "t_half_slow",
    "tau_ratio", "minor_component_fraction", "biologically_resolved",
    "n_full_tail",
    "statistically_selected", "component_separated",
    "minor_component_supported", "tail_supported",
    "curve_consistent", "reportable_two_state",
    "R2_KM_grouped", "RSS", "AIC", "AICc", "BIC",
    "km_unique_points",
]


def single_mle_row(region, avg_n, std_n, n_full, n_cens, n_arr, full, s_fit,
                   model_chosen="none", selection_status="no_fit",
                   delta_BIC=float("nan"), fit_status="fitted",
                   optimizer_status="single_c_mle",
                   r2_km_grouped=float("nan"),
                   fit_counts=None):
    base = {"Region": region, "avg_count": avg_n, "std_count": std_n,
            "n_full": n_full, "n_cens": n_cens, "n_arrivals": n_arr,
            "fit_status": fit_status,
            "model_chosen": model_chosen, "selection_status": selection_status,
            "delta_BIC": delta_BIC,
            "optimizer_status": optimizer_status if s_fit is not None
                                else "not_run_insufficient_data"}
    if fit_counts is not None:
        base.update({
            "fit_event_count": fit_counts.get("fit_event_count", ""),
            "full_fit_count": fit_counts.get("full_fit_count", ""),
            "cens_fit_count": fit_counts.get("cens_fit_count", ""),
            "subsample_seed": fit_counts.get("subsample_seed", ""),
        })
    if s_fit is None:
        return base
    base.update({
        "alpha": s_fit["alpha"], "c": s_fit["c"],
        "tau": s_fit["tau"], "perr_tau": s_fit["perr_tau"],
        "k_off": s_fit["k_off"],
        "mean_dwell_ns": float(np.mean(full)) if n_full else "",
        "median_dwell_ns": float(np.median(full)) if n_full else "",
        # Single-exp mobile residence time = τ (only one finite component).
        "mobile_res_time": float(s_fit["tau"]),
        "R2_KM_eventpoints": s_fit.get("r2_km", float("nan")),
        "R2_KM_grouped": r2_km_grouped,
        "AIC": s_fit["AIC"], "AICc": s_fit["AICc"], "BIC": s_fit["BIC"],
        "nll": s_fit["nll"],
    })
    return base


# Legacy alias name preserved so existing readers (replot, summary) work.
def single_row(region, avg_n, std_n, n_full, n_cens, n_arr, full, s_fit,
               **kw):
    return single_mle_row(region, avg_n, std_n, n_full, n_cens, n_arr,
                          full, s_fit, **kw)


def single_km_ls_row(region, avg_n, std_n, n_full, n_cens, n_arr, s_fit,
                     model_chosen="none", selection_status="no_fit",
                     delta_BIC=float("nan"), fit_status="fitted",
                     km_unique_points=0):
    base = {"Region": region, "avg_count": avg_n, "std_count": std_n,
            "n_full": n_full, "n_cens": n_cens, "n_arrivals": n_arr,
            "fit_status": fit_status,
            "model_chosen": model_chosen,
            "selection_status": selection_status,
            "delta_BIC": delta_BIC,
            "km_unique_points": km_unique_points}
    if s_fit is None:
        return base
    base.update({
        "alpha": s_fit["alpha"], "tau": s_fit["tau"],
        "perr_tau": s_fit["perr_tau"], "c": s_fit["c"],
        "perr_c": s_fit["perr_c"],
        "apparent_res_time": s_fit.get("apparent_res_time", float("nan")),
        # Single-exp mobile residence time = τ.
        "mobile_res_time": float(s_fit["tau"]),
        "R2_KM_grouped": s_fit.get("r_squared", float("nan")),
        "RSS": s_fit.get("rss", float("nan")),
        "AIC": s_fit.get("aic", float("nan")),
        "AICc": s_fit.get("aicc", float("nan")),
        "BIC": s_fit.get("bic", float("nan")),
    })
    return base


def biexp_km_ls_row(region, avg_n, std_n, n_full, n_cens, n_arr, full, b_fit,
                    model_chosen="none", selection_status="no_fit",
                    delta_BIC=float("nan"), fit_status="fitted",
                    km_unique_points=0):
    base = {"Region": region, "avg_count": avg_n, "std_count": std_n,
            "n_full": n_full, "n_cens": n_cens, "n_arrivals": n_arr,
            "fit_status": fit_status,
            "model_chosen": model_chosen,
            "selection_status": selection_status,
            "delta_BIC": delta_BIC,
            "km_unique_points": km_unique_points}
    if b_fit is None:
        return base
    a1, a2 = float(b_fit["alpha1"]), float(b_fit["alpha2"])
    t1, t2 = float(b_fit["tau1"]), float(b_fit["tau2"])
    _, _, resolved = biological_resolved(a1, a2, t1, t2, delta_BIC)
    flags = biological_flags(
        a1, a2, t1, t2, delta_BIC,
        full_dwells=full,
        r2_km_grouped=b_fit.get("r_squared", float("nan")),
        fit_kind="KM-LS")
    base.update({
        "alpha1": a1, "u": b_fit["u"],
        "tau1": t1, "perr_tau1": b_fit["perr_tau1"],
        "alpha2": a2,
        "tau2": t2, "perr_tau2": b_fit["perr_tau2"],
        "c": b_fit["c"], "perr_c": b_fit["perr_c"],
        "fitted_res_time": b_fit.get("fitted_res_time", float("nan")),
        "mobile_res_time": (b_fit.get("mobile_res_time")
                            if b_fit.get("mobile_res_time") is not None
                            else _mobile_rt(a1, t1, a2, t2)),
        "apparent_res_time": b_fit.get("apparent_res_time",
                                        float(np.mean(full)) if n_full else ""),
        "t_half_fast": b_fit.get("t_half_fast", float("nan")),
        "t_half_slow": b_fit.get("t_half_slow", float("nan")),
        "tau_ratio": flags["tau_ratio"],
        "minor_component_fraction": flags["minor_component_fraction"],
        "biologically_resolved": resolved,
        "n_full_tail": flags["n_full_tail"],
        "statistically_selected": flags["statistically_selected"],
        "component_separated": flags["component_separated"],
        "minor_component_supported": flags["minor_component_supported"],
        "tail_supported": flags["tail_supported"],
        "curve_consistent": flags["curve_consistent"],
        "reportable_two_state": flags["reportable_two_state"],
        "R2_KM_grouped": b_fit.get("r_squared", float("nan")),
        "RSS": b_fit.get("rss", float("nan")),
        "AIC": b_fit.get("aic", float("nan")),
        "AICc": b_fit.get("aicc", float("nan")),
        "BIC": b_fit.get("bic", float("nan")),
    })
    return base


def biexp_mle_row(region, avg_n, std_n, n_full, n_cens, n_arr, full, b_fit,
                  rates=None, dataset_meta=None,
                  model_chosen=None, selection_status="no_fit",
                  delta_BIC=float("nan"), fit_status="fitted",
                  r2_km_grouped=float("nan"),
                  fit_counts=None):
    if b_fit is None:
        base = {"Region": region, "avg_count": avg_n, "std_count": std_n,
                "n_full": n_full, "n_cens": n_cens, "n_arrivals": n_arr,
                "fit_status": fit_status,
                "model_chosen": model_chosen or "none",
                "selection_status": selection_status,
                "delta_BIC": delta_BIC,
                "optimizer_status": "not_run_insufficient_data"}
        # rates/dataset metadata still meaningful even when bi failed.
        if rates is not None:
            base["lambda_on_visiting"] = rates.get("lambda_on_visiting", "")
            base["lambda_off"] = rates.get("lambda_off", "")
            base["on_rate_scope"] = rates.get("on_rate_scope", "")
            base["n_arr_total"] = rates.get("n_arr_total", "")
            base["n_diss_total"] = rates.get("n_diss_total", "")
            base["T_free_total_ns"] = rates.get("T_free_total_ns", "")
            base["T_bound_total_ns"] = rates.get("T_bound_total_ns", "")
            base["n_molecules_visiting"] = rates.get("n_molecules_visiting", "")
            base["dt_ns"] = rates.get("dt_ns", "")
            base["n_contact_frames"] = rates.get("n_contact_frames", "")
        if fit_counts is not None:
            base.update({
                "fit_event_count": fit_counts.get("fit_event_count", ""),
                "full_fit_count": fit_counts.get("full_fit_count", ""),
                "cens_fit_count": fit_counts.get("cens_fit_count", ""),
                "subsample_seed": fit_counts.get("subsample_seed", ""),
            })
        return base
    model = model_chosen or b_fit.get("model_chosen", "biexp_c")
    alpha1 = b_fit["alpha_fast"]
    alpha2 = b_fit["alpha_slow"]
    c_const = b_fit["c"]
    finite_mass = alpha1 + alpha2
    u = b_fit.get("u", alpha1 / finite_mass if finite_mass > 0 else float("nan"))
    k_off_mean = (1.0 / b_fit["mean_dwell_ns"]
                  if b_fit["mean_dwell_ns"] and np.isfinite(b_fit["mean_dwell_ns"])
                  else float("nan"))
    dynamic_area = alpha1 * b_fit["tau_fast"] + alpha2 * b_fit["tau_slow"]
    if dynamic_area > 0 and np.isfinite(dynamic_area):
        pop_alpha1 = alpha1 * b_fit["tau_fast"] / dynamic_area
        pop_alpha2 = alpha2 * b_fit["tau_slow"] / dynamic_area
    else:
        pop_alpha1 = pop_alpha2 = float("nan")
    kf = b_fit["k_off_fast"]
    ks = b_fit["k_off_slow"]
    k_off_initial_hazard = alpha1 * kf + alpha2 * ks
    k_off_population = (pop_alpha1 * kf + pop_alpha2 * ks
                        if np.isfinite(pop_alpha1) and np.isfinite(pop_alpha2)
                        else float("nan"))
    _, _, resolved = biological_resolved(
        alpha1, alpha2, b_fit["tau_fast"], b_fit["tau_slow"], delta_BIC)
    flags = biological_flags(
        alpha1, alpha2, b_fit["tau_fast"], b_fit["tau_slow"], delta_BIC,
        full_dwells=full,
        r2_km_grouped=r2_km_grouped,
        fit_kind="MLE")
    row = {
        "Region": region,
        "avg_count": avg_n, "std_count": std_n,
        "n_full": n_full, "n_cens": n_cens, "n_arrivals": n_arr,
        "fit_status": fit_status,
        "model_chosen": model,
        "selection_status": selection_status,
        "delta_BIC": delta_BIC,
        "alpha1": alpha1, "u": u,
        "tau1": b_fit["tau_fast"], "perr_tau1": b_fit["perr_tau_fast"],
        "alpha2": alpha2,
        "tau2": b_fit["tau_slow"], "perr_tau2": b_fit["perr_tau_slow"],
        "c": c_const, "perr_c": b_fit.get("perr_c", float("nan")),
        "event_alpha1": alpha1,
        "event_alpha2": alpha2,
        "event_c": c_const,
        "pop_alpha1_finite": pop_alpha1,
        "pop_alpha2_finite": pop_alpha2,
        "k_off_fast": b_fit["k_off_fast"],
        "k_off_slow": b_fit["k_off_slow"],
        "k_off_mean": k_off_mean,
        "fitted_res_time": b_fit["mean_dwell_ns"],
        "mobile_res_time": _mobile_rt(alpha1, b_fit["tau_fast"],
                                       alpha2, b_fit["tau_slow"]),
        "apparent_res_time": float(np.mean(full)) if n_full else "",
        "t_half_fast": b_fit["t_half_fast"],
        "t_half_slow": b_fit["t_half_slow"],
        "tau_ratio": flags["tau_ratio"],
        "minor_component_fraction": flags["minor_component_fraction"],
        "biologically_resolved": resolved,
        "n_full_tail": flags["n_full_tail"],
        "statistically_selected": flags["statistically_selected"],
        "component_separated": flags["component_separated"],
        "minor_component_supported": flags["minor_component_supported"],
        "tail_supported": flags["tail_supported"],
        "curve_consistent": flags["curve_consistent"],
        "reportable_two_state": flags["reportable_two_state"],
        "R2_KM_eventpoints": b_fit.get("r2_km", float("nan")),
        "R2_KM_grouped": r2_km_grouped,
        "AIC": b_fit["AIC"], "AICc": b_fit["AICc"], "BIC": b_fit["BIC"],
        "nll": b_fit["nll"],
        "nll_pure_bi": b_fit.get("nll_pure_bi", float("nan")),
        "optimizer_status": b_fit.get("optimizer_status", "biexp_c_mle"),
    }
    if fit_counts is not None:
        row.update({
            "fit_event_count": fit_counts.get("fit_event_count", ""),
            "full_fit_count": fit_counts.get("full_fit_count", ""),
            "cens_fit_count": fit_counts.get("cens_fit_count", ""),
            "subsample_seed": fit_counts.get("subsample_seed", ""),
        })
    # Per-molecule hazards from binary contacts[t, m] — independent of the fit
    if rates is not None:
        lam_on_v = rates["lambda_on_visiting"]
        row["lambda_on_visiting"] = lam_on_v
        row["lambda_off"] = rates["lambda_off"]
        row["on_rate_scope"] = rates.get("on_rate_scope", "visiting_molecules")
        row["n_arr_total"] = rates["n_arr_total"]
        row["n_diss_total"] = rates["n_diss_total"]
        row["T_free_total_ns"] = rates["T_free_total_ns"]
        row["T_bound_total_ns"] = rates["T_bound_total_ns"]
        row["n_molecules_visiting"] = rates["n_molecules_visiting"]
        row["dt_ns"] = rates.get("dt_ns", "")
        row["n_contact_frames"] = rates.get("n_contact_frames", "")
        # Per-mode partitioning of the visiting-scope on-hazard.
        row["lambda_on_visiting_fast"] = alpha1 * lam_on_v
        row["lambda_on_visiting_slow"] = alpha2 * lam_on_v
        row["lambda_on_visiting_c"] = c_const * lam_on_v
        row["k_off_population"] = k_off_population
        row["k_off_initial_hazard"] = k_off_initial_hazard
        # Per-mode equilibrium ratios using visiting-scope on-hazard.
        # NOT a true bulk K_eq; the "_visiting" suffix flags the scope.
        if kf > 0:
            row["K_eq_visiting_fast"] = (alpha1 * lam_on_v) / kf
        if ks > 0:
            row["K_eq_visiting_slow"] = (alpha2 * lam_on_v) / ks
        kf_v = row.get("K_eq_visiting_fast", "")
        ks_v = row.get("K_eq_visiting_slow", "")
        if kf_v != "" and ks_v != "":
            row["K_eq_visiting_total_finite"] = kf_v + ks_v
            row["bound_free_ratio_model_finite"] = row["K_eq_visiting_total_finite"]
        total_time = rates["T_free_total_ns"] + rates["T_bound_total_ns"]
        if total_time > 0:
            row["bound_fraction_observed"] = rates["T_bound_total_ns"] / total_time
        if rates["T_free_total_ns"] > 0:
            row["bound_free_ratio_observed"] = (
                rates["T_bound_total_ns"] / rates["T_free_total_ns"]
            )
    # Macroscopic k_on using bulk-solute denominator (single-site model,
    # pseudo-first-order). N_arrivals / ([L_bulk] × T_traj).
    if dataset_meta is not None and rates is not None:
        n_sol = dataset_meta["n_solute"]
        v_box = dataset_meta["v_box_nm3"]
        n_frames = rates.get("n_contact_frames", 0) or 0
        dt_ns = rates.get("dt_ns", 0.0) or 0.0
        T_traj_ns = float(n_frames) * float(dt_ns)
        cb_M = conc_bulk_M(n_sol, v_box) if v_box and n_sol else float("nan")
        row["bulk_solute"] = dataset_meta["solute"]
        row["n_solute_total"] = n_sol
        row["v_box_nm3"] = v_box
        row["conc_bulk_M"] = cb_M
        row["T_traj_ns"] = T_traj_ns
        if (cb_M and np.isfinite(cb_M) and T_traj_ns > 0
                and rates["n_arr_total"] is not None):
            # k_on [M^-1 s^-1] = N_arr / ([L] × T) with T in seconds
            row["k_on_bulk_M_per_s"] = (
                rates["n_arr_total"] / (cb_M * T_traj_ns * 1e-9)
            )
        else:
            row["k_on_bulk_M_per_s"] = float("nan")
        row["k_on_bulk_assumes"] = "single_site,V_box,T_traj"
    return row


# Legacy alias: replot/build_v4_combined_summary already call biexp_row.
def biexp_row(region, avg_n, std_n, n_full, n_cens, n_arr, full, b_fit,
              **kw):
    return biexp_mle_row(region, avg_n, std_n, n_full, n_cens, n_arr,
                         full, b_fit, **kw)


def write_csv(rows, path, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt_val(r.get(c, "")) for c in cols])


# ── Region descriptions parsed from Method 1's run_settings.log ────────────
def load_region_descriptions(dataset_dir):
    """Parse run_settings.log and return {region_id: 'description string'}."""
    log = os.path.join(dataset_dir, "run_settings.log")
    out = {}
    if not os.path.exists(log):
        return out
    cur = None
    with open(log) as f:
        for line in f:
            line = line.rstrip("\n")
            s = line.strip()
            if s.startswith("[") and s.endswith("]"):
                cur = s[1:-1]
                continue
            if cur and s.startswith("Description:"):
                out[cur] = s.split(":", 1)[1].strip()
                cur = None
    return out


# ── Per-region work (pickled to Pool workers) ─────────────────────────────
def _emit_insufficient_data_rows(region, avg_n, std_n, full, cens, n_arr,
                                  rates, dataset_meta, dt_ns,
                                  n_contact_frames, fit_status,
                                  out_dir, desc, plot_dir_mle, plot_dir_ls):
    """V5-4: emit clean insufficient_data rows with NO fitted metrics.

    Used when n_full < MIN_FULL_EVENTS_SINGLE. We still record empirical
    rates (n_arr, λ_off, T_traj, dt) so downstream still has the
    region's existence and arrival counts, but every fit-derived column
    is blank. No plot is generated.

    Mirrors the return tuple shape of `_fit_one_region` so the worker
    pool dispatch stays identical.
    """
    fit_counts = {
        "fit_event_count": 0, "full_fit_count": int(len(full)),
        "cens_fit_count": int(len(cens)), "subsample_seed": 0,
    }
    srow_mle = single_mle_row(
        region, avg_n, std_n, len(full), len(cens), n_arr, full, None,
        model_chosen="none", selection_status="insufficient_data",
        delta_BIC=float("nan"), fit_status=fit_status,
        r2_km_grouped=float("nan"), fit_counts=fit_counts)
    brow_mle = biexp_mle_row(
        region, avg_n, std_n, len(full), len(cens), n_arr, full, None,
        rates=rates, dataset_meta=dataset_meta,
        model_chosen="none", selection_status="insufficient_data",
        delta_BIC=float("nan"), fit_status=fit_status,
        r2_km_grouped=float("nan"), fit_counts=fit_counts)
    srow_ls = single_km_ls_row(
        region, avg_n, std_n, len(full), len(cens), n_arr, None,
        model_chosen="none", selection_status="insufficient_data",
        delta_BIC=float("nan"), fit_status=fit_status,
        km_unique_points=0)
    brow_ls = biexp_km_ls_row(
        region, avg_n, std_n, len(full), len(cens), n_arr, full, None,
        model_chosen="none", selection_status="insufficient_data",
        delta_BIC=float("nan"), fit_status=fit_status,
        km_unique_points=0)
    write_region_summary(
        region, desc, full, cens, n_arr, len(full) + len(cens),
        avg_n, std_n, None, None,
        os.path.join(out_dir, f"summary_{region}.txt"),
        dt_ns=dt_ns, n_contact_frames=n_contact_frames,
        s_ls=None, b_ls=None,
        ls_chosen="none", ls_status="insufficient_data",
        mle_chosen="none", mle_status="insufficient_data",
        km_unique_points=0,
    )
    msg = (f"  {region:>15s}     n_full={len(full):>5}  "
           f"[insufficient_data — no fits]")
    return (region, srow_mle, brow_mle, srow_ls, brow_ls, msg, None)


def _fit_one_region(args):
    """Fit MLE + KM-LS + plot + summarise one region.

    Returns (region, srow_mle, brow_mle, srow_ls, brow_ls, msg, tb).

    Top-level so it can be pickled by multiprocessing. Each call is run in a
    fresh worker (maxtasksperchild=1) to keep matplotlib state clean and
    avoid the SVG-mathtext segfaults seen with long-lived processes.
    """
    (npz, region, desc, out_dir,
     plot_dir_mle, plot_dir_ls, traj_dt_ns, dataset_meta) = args
    try:
        dt_ns, n_contact_frames = contact_timebase(npz, traj_dt_ns=traj_dt_ns)
        # Compute the per-molecule run table once and reuse it for both
        # event extraction and per_molecule_rates. On water protein_shell
        # this saves ~150 s (one full contact_runs pass over 1.5 B records).
        runs = _contact_runs(npz, traj_dt_ns=traj_dt_ns)
        starts, stops, n_bound, dt_runs, n_samples_runs, n_unique = runs
        # Derive completed / right-censored dwells from the cached runs.
        if starts.size == 0:
            full = np.array([], dtype=float); cens = np.array([], dtype=float)
            n_arr = 0; n_total = 0
        else:
            dwells = (stops - starts) * dt_runs
            left_cens = (starts == 0)
            right_cens = (stops == n_samples_runs)
            keep = ~left_cens
            full = dwells[keep & ~right_cens].astype(float)
            cens = dwells[keep & right_cens].astype(float)
            n_total = int(starts.size)
            n_arr = int(np.count_nonzero(keep))
        avg_n, std_n = avg_count_per_frame(npz)
        rates = per_molecule_rates(npz, traj_dt_ns=traj_dt_ns,
                                   cached_runs=runs)

        # ── V5-4: insufficient_data short-circuit ──────────────────────
        # If n_full is below the minimum to fit any model, emit a clean
        # insufficient_data row with NO fitted metrics (rather than running
        # single+c and pretending it was a real selection).
        n_full_real = int(len(full))
        if n_full_real < MIN_FULL_EVENTS_SINGLE:
            fit_status = "insufficient_data"
            return _emit_insufficient_data_rows(
                region, avg_n, std_n, full, cens, n_arr, rates,
                dataset_meta, dt_ns, n_contact_frames, fit_status,
                out_dir, desc, plot_dir_mle, plot_dir_ls)

        # Subsample for fit only (keeps full data for empirical metrics)
        full_fit, cens_fit = _maybe_subsample(full, cens)
        sub_msg = ""
        if len(full_fit) != len(full) or len(cens_fit) != len(cens):
            sub_msg = (f"  [subsample] {region}: fit on "
                       f"{len(full_fit) + len(cens_fit)}/{len(full) + len(cens)} "
                       f"events (full={len(full)}→{len(full_fit)}, "
                       f"cens={len(cens)}→{len(cens_fit)})")
        fit_counts = {
            "fit_event_count": len(full_fit) + len(cens_fit),
            "full_fit_count": len(full_fit),
            "cens_fit_count": len(cens_fit),
            "subsample_seed": 0,
        }

        # ── MLE path: single+c and bi+c via right-censored MLE ─────────
        s_fit = (fit_single_exp_c_mle(full_fit, cens_fit, dt_ns=dt_ns)
                 if len(full_fit) >= MIN_FULL_EVENTS_SINGLE else None)
        b_fit = (fit_bi_exp_c_mle(full_fit, cens_fit, dt_ns=dt_ns)
                 if len(full_fit) >= MIN_FULL_EVENTS_MLE_BI else None)

        # ── KM-LS path: M1-style LS fit on grouped KM survival ─────────
        s_ls = b_ls = None
        t_km, S_km = np.array([0.0]), np.array([1.0])
        if len(full) >= MIN_FULL_EVENTS_LS_BI:
            s_ls, b_ls, t_km, S_km = fit_km_ls(full, cens)

        # R² diagnostics (event-points and grouped KM) for MLE.
        if s_fit is not None:
            s_S = lambda t, a=s_fit["alpha"], tau=s_fit["tau"], c=s_fit["c"]: \
                a * np.exp(-t / tau) + c
            s_fit["r2_km"] = float(r2_against_KM(s_S, full, cens))
            s_fit["r2_km_grouped"] = float(r2_against_KM_grouped(s_S, full, cens))
        if b_fit is not None:
            af, as_, tf, ts, c = (b_fit["alpha_fast"], b_fit["alpha_slow"],
                                  b_fit["tau_fast"], b_fit["tau_slow"],
                                  b_fit["c"])
            b_S = lambda t, af=af, as_=as_, tf=tf, ts=ts, c=c: \
                af * np.exp(-t / tf) + as_ * np.exp(-t / ts) + c
            b_fit["r2_km"] = float(r2_against_KM(b_S, full, cens))
            b_fit["r2_km_grouped"] = float(r2_against_KM_grouped(b_S, full, cens))

        if s_fit is None and b_fit is None and s_ls is None and b_ls is None:
            fit_status = "no_fit"
        else:
            fit_status = "fitted"

        # ── BIC-based model selection — MLE ────────────────────────────
        s_bic_mle = s_fit["BIC"] if s_fit else None
        b_bic_mle = b_fit["BIC"] if b_fit else None
        mle_chosen, mle_status, mle_dbic = select_model_bic(s_bic_mle, b_bic_mle)
        if fit_status == "insufficient_data":
            mle_status = "insufficient_data"
        if b_fit is not None:
            b_fit["model_chosen"] = mle_chosen

        # ── BIC-based model selection — KM-LS ──────────────────────────
        s_bic_ls = s_ls["bic"] if s_ls else None
        b_bic_ls = b_ls["bic"] if b_ls else None
        ls_chosen, ls_status, ls_dbic = select_model_bic(s_bic_ls, b_bic_ls)
        if fit_status == "insufficient_data":
            ls_status = "insufficient_data"

        # Compose a rates dict to overlay on plots (k_off, λ_on, k_on_bulk).
        # rates from per_molecule_rates plus the bulk k_on if we can compute it.
        rates_for_plot = dict(rates) if rates is not None else {}
        if dataset_meta is not None and rates is not None:
            n_sol = dataset_meta.get("n_solute", 0) or 0
            v_box = dataset_meta.get("v_box_nm3", 0) or 0
            n_frames = rates.get("n_contact_frames", 0) or 0
            dt_ns_r = rates.get("dt_ns", 0.0) or 0.0
            T_traj_ns = float(n_frames) * float(dt_ns_r)
            cb_M = conc_bulk_M(n_sol, v_box) if (v_box and n_sol) else float("nan")
            if (cb_M and np.isfinite(cb_M) and T_traj_ns > 0
                    and rates.get("n_arr_total") is not None):
                rates_for_plot["k_on_bulk_M_per_s"] = (
                    float(rates["n_arr_total"]) / (cb_M * T_traj_ns * 1e-9))

        # ── Plots: separate MLE and KM-LS folders ──────────────────────
        # V5-7b: derive per-solute fixed x_max from the dataset name so
        # 0216 / 0311 same-solute plots overlay 1:1.
        dataset_name = os.path.basename(os.path.dirname(plot_dir_mle))
        # plot_dir_mle is ".../v4_<solute>_<run>_method2/plots_mle"
        # so its parent dir basename is the dataset name.
        plot_x_max = _plotting.x_max_for_dataset(dataset_name, default=None)
        plot_fit(region, full, cens, s_fit, b_fit, plot_dir_mle,
                 method_label="MLE", rates=rates_for_plot, x_max=plot_x_max)
        plot_fit(region, full, cens, s_ls, b_ls, plot_dir_ls,
                 method_label="KM-LS", rates=rates_for_plot, x_max=plot_x_max)

        write_region_summary(
            region, desc,
            full, cens, n_arr, n_total, avg_n, std_n,
            s_fit, b_fit,
            os.path.join(out_dir, f"summary_{region}.txt"),
            dt_ns=dt_ns,
            n_contact_frames=n_contact_frames,
            s_ls=s_ls, b_ls=b_ls,
            ls_chosen=ls_chosen, ls_status=ls_status,
            mle_chosen=mle_chosen, mle_status=mle_status,
            km_unique_points=int(len(t_km)),
        )

        srow_mle = single_mle_row(
            region, avg_n, std_n, len(full), len(cens), n_arr, full, s_fit,
            model_chosen=mle_chosen, selection_status=mle_status,
            delta_BIC=mle_dbic, fit_status=fit_status,
            r2_km_grouped=(s_fit.get("r2_km_grouped", float("nan"))
                            if s_fit else float("nan")),
            fit_counts=fit_counts)
        brow_mle = biexp_mle_row(
            region, avg_n, std_n, len(full), len(cens), n_arr, full, b_fit,
            rates=rates, dataset_meta=dataset_meta,
            model_chosen=mle_chosen, selection_status=mle_status,
            delta_BIC=mle_dbic, fit_status=fit_status,
            r2_km_grouped=(b_fit.get("r2_km_grouped", float("nan"))
                            if b_fit else float("nan")),
            fit_counts=fit_counts)
        srow_ls = single_km_ls_row(
            region, avg_n, std_n, len(full), len(cens), n_arr, s_ls,
            model_chosen=ls_chosen, selection_status=ls_status,
            delta_BIC=ls_dbic, fit_status=fit_status,
            km_unique_points=int(len(t_km)))
        brow_ls = biexp_km_ls_row(
            region, avg_n, std_n, len(full), len(cens), n_arr, full, b_ls,
            model_chosen=ls_chosen, selection_status=ls_status,
            delta_BIC=ls_dbic, fit_status=fit_status,
            km_unique_points=int(len(t_km)))

        n_full = len(full); n_cens = len(cens)
        if b_fit is not None:
            c_show = b_fit.get("c", 0.0)
            line = (f"  {region:>15s}     n_full={n_full:7d} n_cens={n_cens:4d}  "
                    f"τ_f={b_fit['tau_fast']:7.3f} τ_s={b_fit['tau_slow']:8.3f} "
                    f"c={c_show:.3f}  R²ₘₗₑ={b_fit.get('r2_km_grouped', float('nan')):.3f} "
                    f"R²ₗₛ={b_ls.get('r_squared', float('nan')) if b_ls else float('nan'):.3f}  "
                    f"[{mle_chosen}/{ls_chosen}]")
        elif s_fit is not None:
            line = (f"  {region:>15s}  n_full={n_full:7d} n_cens={n_cens:4d}  "
                    f"τ={s_fit['tau']:6.3f} (single only)")
        else:
            line = f"  {region:>15s}  n_full={n_full:7d} n_cens={n_cens:4d}  (no fit)"

        msg = "\n".join(filter(None, [sub_msg, line]))
        return (region, srow_mle, brow_mle, srow_ls, brow_ls, msg, None)
    except Exception as e:
        tb = traceback.format_exc()
        return (region, None, None, None, None,
                f"  [error] {region}: {type(e).__name__}: {e}", tb)


# ── Per-dataset driver ─────────────────────────────────────────────────────
def process_dataset(dataset_dir, n_processes=1):
    name = os.path.basename(dataset_dir.rstrip("/"))
    out_dir = method2_output_dir(dataset_dir)
    plot_dir_mle = os.path.join(out_dir, "plots_mle")
    plot_dir_ls = os.path.join(out_dir, "plots_km_ls")
    os.makedirs(plot_dir_mle, exist_ok=True)
    os.makedirs(plot_dir_ls, exist_ok=True)
    # Legacy path kept as a symlink to plots_mle so old callers still resolve.
    legacy_plots = os.path.join(out_dir, "plots")
    if os.path.islink(legacy_plots) or os.path.exists(legacy_plots):
        try:
            if os.path.islink(legacy_plots):
                os.unlink(legacy_plots)
            elif os.path.isdir(legacy_plots):
                # Don't blow away pre-existing real dir; rename it.
                import shutil
                shutil.rmtree(legacy_plots)
        except OSError:
            pass
    try:
        os.symlink("plots_mle", legacy_plots)
    except OSError:
        pass

    cdir = contacts_dir(dataset_dir)
    if not os.path.isdir(cdir):
        print(f"[skip] {name}: no _intermediate/contacts/ dir")
        return

    npz_files = sorted(glob.glob(os.path.join(cdir, "*.npz")))
    if not npz_files:
        print(f"[skip] {name}: no contact npz files")
        return

    descs = load_region_descriptions(dataset_dir)
    traj_dt_ns = load_traj_dt_ns(dataset_dir)
    dataset_meta = DATASET_METADATA.get(name)
    print(f"\n[run] {name} — {len(npz_files)} regions → {out_dir}")
    print(f"  trajectory dt = {traj_dt_ns:.6f} ns/frame", flush=True)
    if dataset_meta is None:
        print(f"  [warn] {name}: no DATASET_METADATA entry — k_on_bulk will be NaN",
              flush=True)
    else:
        cb = conc_bulk_M(dataset_meta["n_solute"], dataset_meta["v_box_nm3"])
        print(f"  bulk: {dataset_meta['solute']} ×{dataset_meta['n_solute']} "
              f"in {dataset_meta['v_box_nm3']:.1f} nm³ → {cb*1000:.1f} mM",
              flush=True)
    t0 = time.perf_counter()

    single_mle_rows, biexp_mle_rows = [], []
    single_ls_rows, biexp_ls_rows = [], []
    n_skip_bi = 0

    skip_existing = os.environ.get("DWELL_V4_SKIP_EXISTING") == "1"
    skip_regions = set(os.environ.get("DWELL_V4_SKIP_REGIONS", "").split(",")) - {""}

    # Seed in-memory rows from any existing CSV so a partial re-run with
    # skip-existing produces a complete merged CSV at the end.
    existing_single_mle = _read_existing_csv(
        os.path.join(out_dir, "method2_mle_single_exp.csv"))
    existing_biexp_mle = _read_existing_csv(
        os.path.join(out_dir, "method2_mle_bi_exp.csv"))
    existing_single_ls = _read_existing_csv(
        os.path.join(out_dir, "method2_km_ls_single_exp.csv"))
    existing_biexp_ls = _read_existing_csv(
        os.path.join(out_dir, "method2_km_ls_bi_exp.csv"))
    seen_in_csv = {r.get("Region") for r in existing_biexp_mle}
    if skip_existing and existing_biexp_mle:
        single_mle_rows.extend(existing_single_mle)
        biexp_mle_rows.extend(existing_biexp_mle)
        single_ls_rows.extend(existing_single_ls)
        biexp_ls_rows.extend(existing_biexp_ls)
        print(f"  [seed] {len(existing_biexp_mle)} rows pre-loaded from existing CSV",
              flush=True)

    # Build the work list, applying skip rules
    todo = []
    for npz in npz_files:
        region = os.path.splitext(os.path.basename(npz))[0]
        if region in skip_regions:
            print(f"  [skip] {region}: in DWELL_V4_SKIP_REGIONS", flush=True)
            continue
        if skip_existing and region in seen_in_csv:
            print(f"  [skip] {region}: already in CSV", flush=True)
            continue
        # If a summary exists but no CSV row, the previous run crashed
        # before write_csv. Reprocess the region so we end up with a fresh
        # summary AND a CSV row — orphan summaries are not a recoverable
        # state.
        todo.append((npz, region, descs.get(region, ""), out_dir,
                     plot_dir_mle, plot_dir_ls, traj_dt_ns, dataset_meta))

    n_workers = max(1, min(int(n_processes), 8, len(todo) or 1))
    # V5-W: ALWAYS use a spawn Pool with maxtasksperchild=1, even for -j 1.
    # Running multiple regions in the same Python process accumulates numpy
    # heap state that triggers C-level SIGSEGVs in `np.sum`/`np.copy` after
    # 2-3 huge regions (water cav03 was the consistent failure point).
    # A fresh subprocess per region resets numpy's internal state and
    # also bounds matplotlib's mathtext state.
    if len(todo) > 0:
        ctx = get_context("spawn")
        with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
            for (region, srow_mle, brow_mle, srow_ls, brow_ls,
                 msg, tb) in pool.imap_unordered(_fit_one_region, todo):
                if msg: print(msg, flush=True)
                if tb: print(tb, flush=True)
                if srow_mle is not None: single_mle_rows.append(srow_mle)
                if brow_mle is not None: biexp_mle_rows.append(brow_mle)
                if srow_ls is not None: single_ls_rows.append(srow_ls)
                if brow_ls is not None: biexp_ls_rows.append(brow_ls)
                if (brow_mle is None or brow_mle.get("model_chosen")
                        in (None, "single_c", "none")):
                    n_skip_bi += 1

    # Sort rows by Region (cav01..cav25, protein_shell last) before writing.
    def _region_key(row):
        r = row.get("Region", "")
        return (0, r) if r.startswith("cav") else (1, r)

    # New canonical schema: 4 CSVs split by objective (MLE vs KM-LS).
    write_csv(sorted(single_mle_rows, key=_region_key),
              os.path.join(out_dir, "method2_mle_single_exp.csv"),
              SINGLE_MLE_COLS)
    write_csv(sorted(biexp_mle_rows, key=_region_key),
              os.path.join(out_dir, "method2_mle_bi_exp.csv"),
              BIEXP_MLE_COLS)
    write_csv(sorted(single_ls_rows, key=_region_key),
              os.path.join(out_dir, "method2_km_ls_single_exp.csv"),
              SINGLE_KM_LS_COLS)
    write_csv(sorted(biexp_ls_rows, key=_region_key),
              os.path.join(out_dir, "method2_km_ls_bi_exp.csv"),
              BIEXP_KM_LS_COLS)
    # Legacy aliases — keep so external readers do not need to change.
    write_csv(sorted(single_mle_rows, key=_region_key),
              os.path.join(out_dir, "single_exp_fitting_results.csv"),
              SINGLE_MLE_COLS)
    write_csv(sorted(biexp_mle_rows, key=_region_key),
              os.path.join(out_dir, "bi_exp_fitting_results.csv"),
              BIEXP_MLE_COLS)

    # Run settings copy + a tiny summary.txt
    src_log = os.path.join(dataset_dir, "run_settings.log")
    if os.path.exists(src_log):
        with open(src_log) as f:
            log_txt = f.read()
        with open(os.path.join(out_dir, "run_settings.log"), "w") as f:
            f.write(log_txt)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(
            "Method 2 — dwell-time MLE + KM-LS on v4 per-region contacts\n"
            f"Dataset: {name}\n"
            f"Source: {dataset_dir}\n"
            f"Regions processed: {len(npz_files)}\n"
            f"Bi-exp MLE fits not preferred (single chosen): {n_skip_bi}\n"
            f"Wall: {time.perf_counter() - t0:.1f} s\n"
            "\n"
            "Files:\n"
            "  method2_mle_single_exp.csv      — MLE single+c per region\n"
            "  method2_mle_bi_exp.csv          — MLE bi+c per region\n"
            "  method2_km_ls_single_exp.csv    — KM-LS single+c per region\n"
            "  method2_km_ls_bi_exp.csv        — KM-LS bi+c per region\n"
            "  single_exp_fitting_results.csv  — alias of method2_mle_single_exp.csv\n"
            "  bi_exp_fitting_results.csv      — alias of method2_mle_bi_exp.csv\n"
            "  summary_<region>.txt            — per-region human-readable\n"
            "  plots_mle/fit_<region>.svg      — MLE survival fit\n"
            "  plots_km_ls/fit_<region>.svg    — KM-LS grouped-KM fit\n"
            "  plots/                          — symlink → plots_mle/\n"
        )

    print(f"  → {len(npz_files)} regions in {time.perf_counter() - t0:.1f}s")


# Backwards-compat shim — discover_datasets is now in kinetics.datasets.
discover_datasets = discover_v4_datasets


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=None,
                    help="Process only the given v4 dataset (e.g. v4_adp_0216). "
                         "If omitted, processes all v4_* under results_v4/.")
    ap.add_argument("-j", "--processes", type=int, default=1,
                    help="Number of worker processes (max 8). Each region "
                         "is fit + plotted in its own subprocess so matplotlib "
                         "state never accumulates. Default 1 (sequential).")
    args = ap.parse_args()

    if args.dataset:
        ds = os.path.join(V4_ROOT, args.dataset)
        if not os.path.isdir(ds):
            print(f"ERROR: {ds} not found", file=sys.stderr)
            sys.exit(1)
        process_dataset(ds, n_processes=args.processes)
    else:
        for ds in discover_datasets():
            process_dataset(ds, n_processes=args.processes)


if __name__ == "__main__":
    sys.exit(main())
