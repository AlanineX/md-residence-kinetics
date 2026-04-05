"""Example configuration for survival probability extraction.

Copy this file to config_extract.py and edit paths/settings for your system.
"""

import os

# ================== TRAJECTORY ==================
TOP_PATH  = "topology.pdb"        # PDB, GRO, or TPR
TRAJ_PATH = "trajectory.xtc"      # XTC or TRR
OUT_DIR   = "results"

# ================== FRAME RANGE ==================
START_FRAME = 0
STOP_FRAME  = None   # exclusive; None = end of trajectory

# ================== ANALYSIS PARAMETERS ==================
INTERMITTENCY          = 0       # allow molecule to leave for N frames
COUNT_STATS_MAX_FRAMES = 500     # frames for occupancy estimate (0 = skip)
FIRST_SHELL_A          = 3.5     # distance cutoff in angstrom
DO_EXP_FIT             = True    # fit single + bi-exponential decay
N_PROCS                = 4       # worker processes for multiprocessing
N_BLOCKS               = 5       # trajectory blocks for block averaging
WATER_O_SELECTION      = "name OW OH2"  # OW for GROMACS, OH2 for CHARMM

# ================== PROTEIN SHELL (global) ==================
CALC_PROTEIN_SHELL     = True    # water SP around entire protein
PROTEIN_SHELL_SETTINGS = {
    "time_resolution_ns": 0.01,  # analysis time step
    "tau_max_ns": 5.0,           # max lag time (must be < trajectory/2)
    "t0_spacing_ns": 0.5,        # spacing between time origins
}

# ================== PER-RESIDUE SOLVATION ==================
# False/None = skip, True = all residues, "0:49" = position range
CALC_PER_RESIDUE_SHELL = None
PER_RESIDUE_SETTINGS = {
    "time_resolution_ns": 0.01,
    "tau_max_ns": 5.0,
    "t0_spacing_ns": 0.5,
    "n_blocks": 1,
}

# ================== PLOT SETTINGS ==================
DO_PLOT            = True
PLOT_X_MAX         = 5.0
PLOT_N_BINS        = 50
PLOT_BIN_SPACING   = 0.5

# ================== TARGET CONFIGS ==================
# Each dict defines a region to analyze. Examples:

# Water near specific residues
# TARGET_CONFIGS = [
#     {"name": "pocket_water", "probe_type": "water",
#      "resid_ranges": ["52 53 87 89 90"],
#      "time_resolution_ns": 0.01, "tau_max_ns": 2.0,
#      "t0_spacing_ns": 0.1, "n_blocks": 5,
#      "description": "Water in binding pocket"},
# ]

# Solute/ligand near protein
# TARGET_CONFIGS = [
#     {"name": "ADP_shell", "probe_type": "solute", "resnames": ["ADP"],
#      "time_resolution_ns": 0.1, "tau_max_ns": 30.0,
#      "t0_spacing_ns": 1.0, "n_blocks": 5,
#      "description": "ADP near protein surface"},
# ]

TARGET_CONFIGS = []
