#!/usr/bin/env python3
"""Standalone fitting + plotting from existing SP CSV files.

Usage:
    cd scripts_kinetics && python run_plot.py
    # or from project root:
    python -m scripts_kinetics.run_plot
"""

import os
import sys
import glob

# Allow running directly from scripts_kinetics/
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from . import config_plot as cfg
    from .fitting import fit_single_exp, fit_bi_exp
    from .plotting import plot_sp_and_fits
    from .utils import write_aggregate_csvs
except ImportError:
    import config_plot as cfg
    from fitting import fit_single_exp, fit_bi_exp
    from plotting import plot_sp_and_fits
    from utils import write_aggregate_csvs

import numpy as np


def main():
    os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(cfg.SOURCE_PATH, "sp_*.csv")))
    if not csv_files:
        print(f"No sp_*.csv files found in {cfg.SOURCE_PATH}")
        return

    print(f"Found {len(csv_files)} CSV files in {cfg.SOURCE_PATH}")

    all_fit_results = []  # (name, fit1, fit2)

    for csv_path in csv_files:
        name = os.path.basename(csv_path).replace("sp_", "").replace(".csv", "")
        try:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            if data.ndim != 2 or data.shape[1] != 2:
                print(f"[{name}] Invalid CSV format. Skipping.")
                continue
            tau_ns = data[:, 0]
            S = data[:, 1]
        except Exception as e:
            print(f"[{name}] Error reading {csv_path}: {e}. Skipping.")
            continue

        print(f"\nProcessing {name} ...")

        # Fit
        fit1 = fit_single_exp(tau_ns, S)
        fit2 = fit_bi_exp(tau_ns, S)

        if fit1 is not None:
            print(f"  1-exp: tau={fit1['tau']:.4f} ns, c={fit1['c']:.4f}, "
                  f"alpha={fit1['alpha']:.4f} | "
                  f"R2={fit1['r_squared']:.4f}, "
                  f"AIC={fit1['aic']:.2f}, AICc={fit1['aicc']:.2f}, BIC={fit1['bic']:.2f}")
        else:
            print("  1-exp fit failed.")

        if fit2 is not None:
            print(f"  2-exp: alpha1={fit2['alpha1']:.4f}, tau1={fit2['tau1']:.4f}, "
                  f"alpha2={fit2['alpha2']:.4f}, tau2={fit2['tau2']:.4f}, "
                  f"c={fit2['c']:.4f} | "
                  f"R2={fit2['r_squared']:.4f}, "
                  f"AIC={fit2['aic']:.2f}, AICc={fit2['aicc']:.2f}, BIC={fit2['bic']:.2f}")
            print(f"    Fitted residence time: {fit2['fitted_res_time']:.4f} ns")
            if fit2["apparent_res_time"] is not None:
                print(f"    Apparent residence time: {fit2['apparent_res_time']:.4f} ns")
            if fit2["t_half_overall"] is not None:
                print(f"    Overall half-life: {fit2['t_half_overall']:.4f} ns")
        else:
            print("  2-exp fit failed.")

        # Plot (optional)
        if getattr(cfg, "do_plot", True):
            plot_sp_and_fits(
                tau_ns, S, fit1, fit2,
                name=name, outdir=cfg.OUTPUT_PATH,
                x_max_plot=cfg.x_max_plot,
                n_bins=cfg.n_bins,
                bin_spacing_factor=cfg.bin_spacing_factor,
                base_fontsize=cfg.base_fontsize,
            )
            print(f"  Plots saved to {cfg.OUTPUT_PATH}")

        all_fit_results.append((name, fit1, fit2))

    # Aggregate CSVs
    write_aggregate_csvs(all_fit_results, cfg.OUTPUT_PATH)
    print(f"\nDone. Results in: {cfg.OUTPUT_PATH}")


if __name__ == "__main__":
    main()
