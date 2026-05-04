#!/usr/bin/env python3
"""Extract survival probability from MD trajectory, fit, and plot.

Three-phase pipeline (see kinetics package for the implementation):
  1. Phase A: build per-frame contact records for ALL regions
     (each region uses N_PROCS cores for frame-chunk parallelism)
  2. Phase B: compute per-t0 SP contributions for ALL regions
     (single-process per region — fast enough on contact records)
  3. Phase C: aggregate, fit, plot, write CSVs and summaries

Usage:
    cd scripts_kinetics && python run_pipeline.py
    # or from project root:
    python -m scripts_kinetics.run_pipeline
"""

import os
import sys
import time
import warnings

# Allow running directly from scripts_kinetics/
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from . import config_extract as cfg
    from .kinetics import (
        dispatch_phase_a_subprocess,
        compute_sp_for_region,
        aggregate_sp_for_region,
        load_contacts_npz,
        _contacts_path,
        _sp_origins_path,
    )
    from .fitting import fit_single_exp, fit_bi_exp, model_free_metrics
    from .plotting import plot_sp_and_fits
    from .utils import (
        make_regions, validate_and_compute_settings,
        save_sp_csv, write_run_log, write_summary, write_aggregate_csvs,
    )
except ImportError:
    import config_extract as cfg
    from kinetics import (
        dispatch_phase_a_subprocess,
        compute_sp_for_region,
        aggregate_sp_for_region,
        load_contacts_npz,
        _contacts_path,
        _sp_origins_path,
    )
    from fitting import fit_single_exp, fit_bi_exp, model_free_metrics
    from plotting import plot_sp_and_fits
    from utils import (
        make_regions, validate_and_compute_settings,
        save_sp_csv, write_run_log, write_summary, write_aggregate_csvs,
    )

import numpy as np


def _elapsed(t0):
    """Format elapsed time since t0."""
    s = time.perf_counter() - t0
    if s < 60:
        return f"{s:.1f}s"
    return f"{s/60:.1f}min"


def _avg_residues_from_contacts(contacts_path):
    """Compute mean/std of contact-set sizes (used for the summary log)."""
    if not os.path.exists(contacts_path):
        return float("nan"), float("nan"), 0
    _, list_of_sets = load_contacts_npz(contacts_path)
    sizes = np.array([len(s) for s in list_of_sets], dtype=np.float64)
    if sizes.size == 0:
        return float("nan"), float("nan"), 0
    return float(sizes.mean()), float(sizes.std()), int(sizes.size)


def main():
    t_start = time.perf_counter()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # ── Setup ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    regions, u = make_regions(
        cfg.TOP_PATH, cfg.TRAJ_PATH, cfg.TARGET_CONFIGS, cfg.OUT_DIR,
        cutoff_a=cfg.FIRST_SHELL_A,
        water_o_selection=cfg.WATER_O_SELECTION,
        calc_protein_shell=cfg.CALC_PROTEIN_SHELL,
        protein_shell_settings=getattr(cfg, "PROTEIN_SHELL_SETTINGS", None),
        calc_per_residue_shell=getattr(cfg, "CALC_PER_RESIDUE_SHELL", None),
        per_residue_settings=getattr(cfg, "PER_RESIDUE_SETTINGS", None),
    )
    print(f"[timer] Universe + regions loaded in {_elapsed(t0)}")

    # Trajectory info
    dt_ps = getattr(u.trajectory, "dt", None)
    if dt_ps is None:
        warnings.warn("Trajectory dt not set; assuming 10 ps per frame.")
        dt_ps = 10.0
    dt_ns = float(dt_ps) / 1000.0
    n_frames_total = len(u.trajectory)
    start = max(0, cfg.START_FRAME or 0)
    stop = min(n_frames_total, cfg.STOP_FRAME) if cfg.STOP_FRAME is not None else n_frames_total

    # Validate and compute frame-based settings
    t0 = time.perf_counter()
    validate_and_compute_settings(regions, dt_ns, stop - start)
    print(f"[timer] Validation in {_elapsed(t0)}")

    # Write run log
    write_run_log(
        regions, cfg.OUT_DIR, cfg.TOP_PATH, cfg.TRAJ_PATH,
        cfg.START_FRAME, cfg.STOP_FRAME, cfg.INTERMITTENCY,
        cfg.FIRST_SHELL_A,
        cfg.WATER_O_SELECTION, cfg.DO_EXP_FIT, cfg.N_PROCS,
        dt_ns=dt_ns, n_frames=n_frames_total,
    )

    # Note: keep the main-process Universe alive — closing it can crash MDA

    # ── Resume detection ─────────────────────────────────────────────────────
    regions_to_process = []
    skipped_regions = []
    for region in regions:
        if (os.path.exists(region.csv_path)
                and os.path.getsize(region.csv_path) > 0):
            skipped_regions.append(region)
        else:
            regions_to_process.append(region)

    if skipped_regions:
        print(f"\n[resume] Skipping {len(skipped_regions)} regions with existing CSVs.")

    if not regions_to_process:
        print("\nAll regions already done. Aggregating from existing CSVs only.")

    # ── Phase A: contacts for all regions (parallel via subprocess pool) ─────
    print("\n" + "=" * 70)
    print(f"PHASE A — Contact recording ({len(regions_to_process)} regions, "
          f"max {cfg.N_PROCS} parallel subprocesses)")
    print("=" * 70)
    phase_a_t0 = time.perf_counter()

    # Filter out regions whose contacts already exist
    needs_phase_a = []
    for region in regions_to_process:
        contacts_path = _contacts_path(cfg.OUT_DIR, region.name)
        n_expected = len(range(start, stop, region.stride))
        if os.path.exists(contacts_path):
            try:
                data = np.load(contacts_path)
                if len(data["frame_indices"]) == n_expected:
                    print(f"[Phase A skip] {region.name}: contacts already exist "
                          f"({n_expected} frames)")
                    continue
            except Exception:
                pass
        needs_phase_a.append(region)

    if needs_phase_a:
        phase_a_times = dispatch_phase_a_subprocess(
            needs_phase_a, cfg.TOP_PATH, cfg.TRAJ_PATH, cfg.OUT_DIR,
            start, stop, cfg.N_PROCS,
        )
    else:
        phase_a_times = {}

    phase_a_total = time.perf_counter() - phase_a_t0
    print(f"\n[timer] Phase A total: {_elapsed(phase_a_t0)}")

    # ── Phase B: per-t0 SP for all regions ───────────────────────────────────
    print("\n" + "=" * 70)
    print(f"PHASE B — SP from contacts ({len(regions_to_process)} regions)")
    print("=" * 70)
    phase_b_t0 = time.perf_counter()
    phase_b_times = {}
    for i, region in enumerate(regions_to_process):
        print(f"\n-- Phase B {i+1}/{len(regions_to_process)}: {region.name} --")
        _, elapsed = compute_sp_for_region(
            region, cfg.OUT_DIR, cfg.INTERMITTENCY, cfg.N_PROCS,
        )
        phase_b_times[region.name] = elapsed
    phase_b_total = time.perf_counter() - phase_b_t0
    print(f"\n[timer] Phase B total: {_elapsed(phase_b_t0)}")

    # ── Phase C: aggregate, fit, plot, summary ───────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE C — Aggregate + fit + plot + summary")
    print("=" * 70)
    phase_c_t0 = time.perf_counter()

    summary_results = []
    all_fit_results = []
    count_data = {}
    mf_data = {}

    # First handle skipped (already-done) regions: load CSVs for aggregate
    for region in skipped_regions:
        try:
            data = np.loadtxt(region.csv_path, delimiter=",", skiprows=1)
            t_arr, S_arr = data[:, 0], data[:, 1]
            fit1 = fit_single_exp(t_arr, S_arr) if cfg.DO_EXP_FIT else None
            fit2 = fit_bi_exp(
                t_arr, S_arr,
                tau_floor=getattr(cfg, "BI_EXP_TAU_FLOOR", None),
                tau_ceil=getattr(cfg, "BI_EXP_TAU_CEIL", None),
                reject_degenerate=getattr(cfg, "BI_EXP_REJECT_DEGENERATE", False),
            ) if cfg.DO_EXP_FIT else None
            mf = model_free_metrics(t_arr, S_arr)
            mf_data[region.name] = mf
            c_est = float(S_arr[-1]) if S_arr.size else 0.0
            tau_res = float(np.trapezoid(S_arr - c_est, t_arr))
            summary_results.append((region.name, tau_res, 0.0))
            all_fit_results.append((region.name, fit1, fit2))
        except Exception as e:
            print(f"  [WARN] Could not load existing CSV for {region.name}: {e}")

    # Now process newly-computed regions
    for region in regions_to_process:
        t_region = time.perf_counter()
        print(f"\n-- Phase C: {region.name} --")

        try:
            taus_frames, S, counts_tot = aggregate_sp_for_region(region, cfg.OUT_DIR)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")
            summary_results.append((region.name, 0.0, 0.0))
            continue

        if not np.any(counts_tot[1:] > 0):
            print(f"  [WARN] No valid SP data for {region.name} — skipping.")
            summary_results.append((region.name, 0.0, 0.0))
            continue

        tau_ns = taus_frames * dt_ns * region.stride

        # Counts (avg/std of contact-set sizes from contacts intermediate)
        contacts_path = _contacts_path(cfg.OUT_DIR, region.name)
        avg_residues, std_residues, n_count = _avg_residues_from_contacts(contacts_path)

        sp_dict = {
            "tau_ns": tau_ns,
            "S": S,
            "counts": counts_tot,
            "avg_residues": avg_residues,
            "std_residues": std_residues,
            "n_count_frames": n_count,
            "count_step": region.stride,
            "dt_ns": dt_ns,
            "start_frame": start,
            "stop_frame": stop,
            "time_taken": phase_a_times.get(region.name, 0.0)
                          + phase_b_times.get(region.name, 0.0),
        }

        # Save SP CSV
        save_sp_csv(tau_ns, S, region.csv_path)

        # Fit
        t0 = time.perf_counter()
        fit1 = fit2 = None
        if cfg.DO_EXP_FIT:
            fit1 = fit_single_exp(tau_ns, S)
            fit2 = fit_bi_exp(
                tau_ns, S,
                tau_floor=getattr(cfg, "BI_EXP_TAU_FLOOR", None),
                tau_ceil=getattr(cfg, "BI_EXP_TAU_CEIL", None),
                reject_degenerate=getattr(cfg, "BI_EXP_REJECT_DEGENERATE", False),
            )
            if fit1 is not None:
                print(f"  1-exp: tau={fit1['tau']:.4f} ns, c={fit1['c']:.4f}, "
                      f"R2={fit1['r_squared']:.4f}")
            else:
                print("  1-exp fit failed.")
            if fit2 is not None:
                print(f"  2-exp: tau1={fit2['tau1']:.4f}, tau2={fit2['tau2']:.4f}, "
                      f"c={fit2['c']:.4f}, R2={fit2['r_squared']:.4f}")
                print(f"    Fitted residence time: {fit2['fitted_res_time']:.4f} ns")
            else:
                print("  2-exp fit failed.")
        print(f"  [timer] Fitting in {_elapsed(t0)}")

        # Summary
        write_summary(region, sp_dict, fit1, fit2, region.txt_path,
                      do_exp_fit=cfg.DO_EXP_FIT)

        # Plot
        if cfg.DO_PLOT:
            t0 = time.perf_counter()
            plot_dir = os.path.join(cfg.OUT_DIR, "plots")
            plot_sp_and_fits(
                tau_ns, S, fit1, fit2,
                name=region.name, outdir=plot_dir,
                x_max_plot=cfg.PLOT_X_MAX,
                n_bins=cfg.PLOT_N_BINS,
                bin_spacing_factor=cfg.PLOT_BIN_SPACING,
                species_label=getattr(cfg, "PLOT_SPECIES_LABEL", None),
            )
            print(f"  [timer] Plotting in {_elapsed(t0)}")

        # Model-free metrics
        mf = model_free_metrics(tau_ns, S)
        mf_data[region.name] = mf
        for k, v in sorted(mf.items()):
            if v is not None:
                print(f"    {k} = {v:.6f}")

        c_est = float(S[-1]) if S.size else 0.0
        tau_res = float(np.trapezoid(S - c_est, tau_ns))
        summary_results.append((region.name, tau_res, sp_dict["time_taken"]))
        all_fit_results.append((region.name, fit1, fit2))
        count_data[region.name] = (avg_residues, std_residues)
        print(f"  [timer] Region C: {_elapsed(t_region)}")

    print(f"\n[timer] Phase C total: {_elapsed(phase_c_t0)}")

    # Aggregate CSVs — only write if we have a meaningful number of regions.
    # When running as a single-region SLURM array task, writing a 1-row
    # aggregate is wasteful and creates a race condition (50 tasks all
    # overwriting the same CSV). The merge job (merge_results.sh) produces
    # the authoritative aggregate after all tasks finish.
    n_total_regions = len(regions_to_process) + len(skipped_regions)
    if all_fit_results and n_total_regions > 1:
        t0 = time.perf_counter()
        write_aggregate_csvs(all_fit_results, cfg.OUT_DIR,
                             count_data=count_data, model_free_data=mf_data)
        print(f"[timer] Aggregate CSVs in {_elapsed(t0)}")
    elif all_fit_results:
        print(f"[skip] Aggregate CSVs skipped (single-region task; merge job will produce them)")

    # Optional cleanup of intermediates
    keep_int = getattr(cfg, "KEEP_INTERMEDIATES", True)
    if not keep_int:
        print("\n[cleanup] Removing _intermediate files (KEEP_INTERMEDIATES=False)")
        for region in regions_to_process:
            for path in (_contacts_path(cfg.OUT_DIR, region.name),
                         _sp_origins_path(cfg.OUT_DIR, region.name)):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    print(f"  [WARN] Could not remove {path}: {e}")

    # ── Final summary + per-region/per-core breakdown ────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, tau_ns, t in summary_results:
        print(f"[{name}] Integral residence time = {tau_ns:.6f} ns ({t:.2f} s)")

    print("\n" + "=" * 70)
    print("TIMING BREAKDOWN")
    print("=" * 70)
    print(f"  Setup + validation: (see [timer] lines above)")
    print(f"  Phase A total:      {phase_a_total:.1f}s")
    print(f"  Phase B total:      {phase_b_total:.1f}s")
    print(f"  Phase C total:      {time.perf_counter() - phase_c_t0:.1f}s")
    if regions_to_process:
        print(f"\nPer-region timings (N_PROCS={cfg.N_PROCS}):")
        print(f"  {'region':<30} {'phaseA(s)':>12} {'phaseB(s)':>12}")
        for r in regions_to_process:
            print(f"  {r.name:<30} {phase_a_times.get(r.name, 0):>12.1f} "
                  f"{phase_b_times.get(r.name, 0):>12.1f}")
    if skipped_regions:
        print(f"\n[resume] Skipped {len(skipped_regions)}/{len(regions)} regions "
              f"with existing CSVs.")

    print(f"\n[timer] Total pipeline: {_elapsed(t_start)}")
    print(f"Done. SP CSVs in: {cfg.OUT_DIR}")


if __name__ == "__main__":
    main()
