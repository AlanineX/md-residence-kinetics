"""Benchmark: protein_shell on 20ns GroEL trajectory.

Tests the full pipeline including block-wise Phase B for large regions.
"""

import os

TOP_PATH  = "/home/alan/working/groel_new/0216_GROEL_ADP/out_1_whole_water_sk1_ref0.pdb"
TRAJ_PATH = "/home/alan/working/groel_new/0216_GROEL_ADP/out_1_whole_water_sk1_20ns.xtc"
OUT_DIR   = "/tmp/benchmark_shell"

START_FRAME = 0
STOP_FRAME  = None
INTERMITTENCY = 0
FIRST_SHELL_A = 3.5
DO_EXP_FIT = True
N_PROCS = 12
WATER_O_SELECTION = "name OW OH2"
CALC_PROTEIN_SHELL = True
CALC_PER_RESIDUE_SHELL = None
PER_RESIDUE_SETTINGS = {}
KEEP_INTERMEDIATES = True

DO_PLOT = False
PLOT_X_MAX = 5.0
PLOT_N_BINS = 50
PLOT_BIN_SPACING = 0.5

PROTEIN_SHELL_SETTINGS = {
    "time_resolution_ns": 0.01,
    "tau_max_ns": 5.0,
    "t0_spacing_ns": 0.5,
}

TARGET_CONFIGS = []
