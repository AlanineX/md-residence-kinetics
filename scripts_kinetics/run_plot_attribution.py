#!/usr/bin/env python3
"""Plot additive attribution from existing CSVs without trajectory extraction."""

import csv
import glob
import os
import sys

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from . import config_plot_attribution as cfg
    from .plotting import plot_attribution
except ImportError:
    import config_plot_attribution as cfg
    from plotting import plot_attribution


def load_attribution_csv(path):
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    required = ["time_ns", "regional_survival", "closure_error"]
    if not all(column in header for column in required):
        raise ValueError(f"{path} is not an attribution curve CSV")
    labels = [column for column in header if column not in required]
    data = np.asarray(rows, dtype=float)
    columns = {column: header.index(column) for column in header}
    components = np.asarray([
        data[:, columns[label]] for label in labels
    ])
    return data[:, columns["time_ns"]], data[:, columns["regional_survival"]], components, labels


def main():
    os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)
    files = sorted(glob.glob(os.path.join(cfg.SOURCE_PATH, "attribution_*.csv")))
    files = [path for path in files
             if not os.path.basename(path).startswith("attribution_integrals_")]
    if not files:
        print(f"No attribution_*.csv files found in {cfg.SOURCE_PATH}")
        return
    for path in files:
        name = os.path.basename(path)[len("attribution_"):-len(".csv")]
        try:
            time_ns, regional, components, labels = load_attribution_csv(path)
            output = plot_attribution(
                time_ns, regional, components, labels, name, cfg.OUTPUT_PATH,
                x_max_plot=cfg.X_MAX_PLOT, n_bins=cfg.N_BINS,
                bin_spacing_factor=cfg.BIN_SPACING_FACTOR,
                base_fontsize=cfg.BASE_FONTSIZE,
                stack_order=cfg.STACK_ORDER,
                rank_horizon_ns=cfg.RANK_HORIZON_NS,
                largest_at_bottom=cfg.LARGEST_AT_BOTTOM,
                color_method=cfg.COLOR_METHOD, palette=cfg.PALETTE,
                component_colors=cfg.COMPONENT_COLORS,
                manual_order=cfg.MANUAL_ORDER, title=cfg.TITLE)
            print(f"[{name}] Saved {output}")
        except Exception as error:
            print(f"[{name}] Error: {error}")


if __name__ == "__main__":
    main()
