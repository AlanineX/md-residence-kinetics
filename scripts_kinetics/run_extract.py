#!/usr/bin/env python3
"""Extract survival probability from MD trajectory, fit, and plot.

Usage:
    cd scripts_kinetics && python run_extract.py
    # or from project root:
    python -m scripts_kinetics.run_extract
"""

import os
import sys
import warnings

# Allow running directly from scripts_kinetics/
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from . import config_extract as cfg
    from .kinetics import compute_sp
    from .fitting import fit_single_exp, fit_bi_exp
    from .plotting import plot_sp_and_fits
    from .utils import (
        make_regions, validate_and_compute_settings,
        save_sp_csv, write_run_log, write_summary, write_aggregate_csvs,
    )
except ImportError:
    import config_extract as cfg
    from kinetics import compute_sp
    from fitting import fit_single_exp, fit_bi_exp
    from plotting import plot_sp_and_fits
    from utils import (
        make_regions, validate_and_compute_settings,
        save_sp_csv, write_run_log, write_summary, write_aggregate_csvs,
    )

import numpy as np


def main():
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # Build regions from target configs (returns shared Universe to avoid re-load crash)
    regions, u = make_regions(
        cfg.TOP_PATH, cfg.TRAJ_PATH, cfg.TARGET_CONFIGS, cfg.OUT_DIR,
        cutoff_a=cfg.FIRST_SHELL_A,
        water_o_selection=cfg.WATER_O_SELECTION,
        calc_protein_shell=cfg.CALC_PROTEIN_SHELL,
        n_blocks_default=cfg.N_BLOCKS,
        protein_shell_settings=getattr(cfg, "PROTEIN_SHELL_SETTINGS", None),
        calc_per_residue_shell=getattr(cfg, "CALC_PER_RESIDUE_SHELL", None),
        per_residue_settings=getattr(cfg, "PER_RESIDUE_SETTINGS", None),
    )

    # Get trajectory info for validation (reuse universe)
    dt_ps = getattr(u.trajectory, "dt", None)
    if dt_ps is None:
        warnings.warn("Trajectory dt not set; assuming 10 ps per frame.")
        dt_ps = 10.0
    dt_ns = float(dt_ps) / 1000.0
    n_frames_total = len(u.trajectory)
    start = max(0, cfg.START_FRAME or 0)
    stop = min(n_frames_total, cfg.STOP_FRAME) if cfg.STOP_FRAME is not None else n_frames_total
    n_frames = stop - start

    # Validate and compute frame-based settings (using effective window)
    validate_and_compute_settings(regions, dt_ns, n_frames)

    # Write run log
    write_run_log(
        regions, cfg.OUT_DIR, cfg.TOP_PATH, cfg.TRAJ_PATH,
        cfg.START_FRAME, cfg.STOP_FRAME, cfg.INTERMITTENCY,
        cfg.COUNT_STATS_MAX_FRAMES, cfg.FIRST_SHELL_A,
        cfg.WATER_O_SELECTION, cfg.DO_EXP_FIT, cfg.N_PROCS,
        dt_ns=dt_ns, n_frames=n_frames_total,
    )

    # Process each region: extract -> fit -> plot
    summary_results = []   # (name, tau_res_ns, time_taken)
    all_fit_results = []   # (name, fit1, fit2) for aggregate CSVs

    for region in regions:
        print(f"\n== Region: {region.name} ==")

        # Step 1: Compute survival probability (pass universe to avoid re-load)
        sp = compute_sp(
            region, cfg.TOP_PATH, cfg.TRAJ_PATH,
            start_frame=cfg.START_FRAME,
            stop_frame=cfg.STOP_FRAME,
            intermittency=cfg.INTERMITTENCY,
            count_stats_max_frames=cfg.COUNT_STATS_MAX_FRAMES,
            n_procs=cfg.N_PROCS,
            universe=u,
        )

        if sp is None:
            # No probe ever visited this region — skip it
            summary_results.append((region.name, 0.0, 0.0))
            continue

        # Step 2: Save SP CSV
        save_sp_csv(sp["tau_ns"], sp["S"], region.csv_path)

        # Step 3: Fit
        fit1 = fit2 = None
        if cfg.DO_EXP_FIT:
            fit1 = fit_single_exp(sp["tau_ns"], sp["S"])
            fit2 = fit_bi_exp(sp["tau_ns"], sp["S"])

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

        # Step 4: Write summary
        write_summary(region, sp, fit1, fit2, region.txt_path,
                      do_exp_fit=cfg.DO_EXP_FIT)

        # Step 5: Plot
        if cfg.DO_PLOT:
            plot_dir = os.path.join(cfg.OUT_DIR, "plots")
            plot_sp_and_fits(
                sp["tau_ns"], sp["S"], fit1, fit2,
                name=region.name, outdir=plot_dir,
                x_max_plot=cfg.PLOT_X_MAX,
                n_bins=cfg.PLOT_N_BINS,
                bin_spacing_factor=cfg.PLOT_BIN_SPACING,
            )
            print(f"  Plots saved to {plot_dir}")

        c_est = float(sp["S"][-1]) if sp["S"].size else 0.0
        tau_res = float(np.trapezoid(sp["S"] - c_est, sp["tau_ns"]))
        summary_results.append((region.name, tau_res, sp["time_taken"]))
        all_fit_results.append((region.name, fit1, fit2))

    # Aggregate CSVs
    if all_fit_results:
        write_aggregate_csvs(all_fit_results, cfg.OUT_DIR)

    # Final summary
    print("\n" + "=" * 60)
    for name, tau_ns, t in summary_results:
        print(f"[{name}] Integral residence time = {tau_ns:.6f} ns ({t:.2f} s)")
    print(f"Done. SP CSVs in: {cfg.OUT_DIR}")


if __name__ == "__main__":
    main()
