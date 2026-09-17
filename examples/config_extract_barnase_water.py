"""Example: chain-specific water residence around every barnase/barstar residue."""

TOP_PATH = "/path/to/barnase_barstar.tpr"
TRAJ_PATH = "/path/to/barnase_barstar.xtc"
OUT_DIR = "/path/to/water_kinetics"

START_FRAME = 0
STOP_FRAME = None

FIRST_SHELL_A = 3.5
WATER_O_SELECTION = "name OW"
INTERMITTENCY = 0
COUNT_STATS_MAX_FRAMES = 500
N_PROCS = 12
N_BLOCKS = 5

CALC_PROTEIN_SHELL = False
CALC_PER_RESIDUE_SHELL = True

PROTEIN_SHELL_SETTINGS = {
    "time_resolution_ns": 0.01,
    "tau_max_ns": 5.0,
    "t0_spacing_ns": 0.5,
}

PER_RESIDUE_SETTINGS = {
    "time_resolution_ns": 0.01,
    "tau_max_ns": 5.0,
    "t0_spacing_ns": 0.5,
    "n_blocks": 5,
}

DO_EXP_FIT = True
DO_PLOT = True
PLOT_X_MAX = 2.0
PLOT_N_BINS = 50
PLOT_BIN_SPACING = 0.5

TARGET_CONFIGS = []
