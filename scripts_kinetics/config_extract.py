"""Configuration for survival probability extraction pipeline.

Edit TOP_PATH / TRAJ_PATH / OUT_DIR for your own data, or set the
KINETICS_TOP / KINETICS_TRAJ / KINETICS_OUT environment variables.
"""

import os

# ================== TRAJECTORY ==================
TOP_PATH  = os.environ.get("KINETICS_TOP",  "/path/to/topology.pdb")
TRAJ_PATH = os.environ.get("KINETICS_TRAJ", "/path/to/trajectory.xtc")
OUT_DIR   = os.environ.get("KINETICS_OUT",  "./kinetics_results")

# ================== FRAME RANGE ==================
START_FRAME = 0
STOP_FRAME  = None   # exclusive; None = end

# ================== ANALYSIS PARAMETERS ==================
INTERMITTENCY          = 0
COUNT_STATS_MAX_FRAMES = 500   # 0 = skip count estimation
FIRST_SHELL_A          = 3.5   # cutoff (angstrom)
DO_EXP_FIT             = True
N_PROCS                = 10
N_BLOCKS               = 1
WATER_O_SELECTION      = "name OW OH2"
CALC_PROTEIN_SHELL     = True # False
KEEP_INTERMEDIATES     = True  # keep _intermediate/contacts and sp_origins after final CSV

# Per-residue solvation: False/None = skip, True = all, "0:49" = positions 0-49
CALC_PER_RESIDUE_SHELL = None
PER_RESIDUE_SETTINGS = {
    "time_resolution_ns": 0.01,
    "tau_max_ns": 5.0,
    "t0_spacing_ns": 0.5,
    "n_blocks": 1,
}

# ================== PLOT SETTINGS (after extraction) ==================
DO_PLOT            = True
PLOT_X_MAX         = 2.0
PLOT_N_BINS        = 50
PLOT_BIN_SPACING   = 0.5

# ================== POCKET RESIDUES (union of 7 sets) ==================
POCKET_RESID_SETS = [
    "52 53 87 89 90 398",
    "600 601 635 637 638 946",
    "1148 1149 1183 1185 1186 1494",
    "1696 1697 1731 1733 1734 2042",
    "2244 2245 2279 2281 2282 2590",
    "2792 2793 2827 2829 2830 3138",
    "3340 3341 3375 3377 3378 3686",
]

# ================== PER-SPECIES SETTINGS (all in ns) ==================
# MDA: fast dynamics (tau ~ 0.3-4 ns)
MDA_SETTINGS = {
    "time_resolution_ns": 0.01,
    "tau_max_ns": 5.0,
    "t0_spacing_ns": 0.5,
    "n_blocks": 1,
}

# Protein shell water settings
PROTEIN_SHELL_SETTINGS = {
    "time_resolution_ns": 0.01,
    "tau_max_ns": 5.0,
    "t0_spacing_ns": 0.5,
}

# DDA: slow dynamics (tau ~ 20-150 ns)
DDA_SETTINGS = {
    "time_resolution_ns": 0.05,
    "tau_max_ns": 250.0,
    "t0_spacing_ns": 5.0,
    "n_blocks": 4,
}

# Water pocket: very fast dynamics (tau ~ 0.01-0.5 ns)
WATER_POCKET_SETTINGS = {
    "time_resolution_ns": 0.002,
    "tau_max_ns": 2.0,
    "t0_spacing_ns": 0.05,
    "n_blocks": 10,
}

# ================== TARGET CONFIGS ==================
TARGET_CONFIGS = [
    # {"name": "pocket_water", "probe_type": "water",
    #  "resid_ranges": POCKET_RESID_SETS,
    #  **WATER_POCKET_SETTINGS,
    #  "description": "Water within cutoff of pocket residues (union of 7 sets)."},

    # {"name": "MDA_shell", "probe_type": "solute", "resnames": ["MDA"],
    #  **MDA_SETTINGS,
    #  "description": "MDA residues within cutoff of protein (global protein shell)."},
    # {"name": "pocket_MDA", "probe_type": "solute", "resnames": ["MDA"],
    #  "resid_ranges": POCKET_RESID_SETS,
    #  **MDA_SETTINGS,
    #  "description": "MDA within cutoff of pocket residues (union of 7 sets)."},

    # {"name": "DDA_shell", "probe_type": "solute", "resnames": ["DDA"],
    #  **DDA_SETTINGS,
    #  "description": "DDA residues within cutoff of protein (global protein shell)."},
    # {"name": "pocket_DDA", "probe_type": "solute", "resnames": ["DDA"],
    #  "resid_ranges": POCKET_RESID_SETS,
    #  **DDA_SETTINGS,
    #  "description": "DDA within cutoff of pocket residues (union of 7 sets)."},
]
