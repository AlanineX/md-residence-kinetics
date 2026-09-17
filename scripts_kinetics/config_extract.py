"""Edit these settings, then run: python run_extract.py"""

TOP_PATH = "/path/to/topology.tpr"       # PDB, GRO, or TPR.
TRAJ_PATH = "/path/to/trajectory.xtc"    # XTC or TRR with matching atoms.
OUT_DIR = "/path/to/results"

START_FRAME = 0                           # First frame; zero-based.
STOP_FRAME = None                         # Exclusive; None means the end.

FIRST_SHELL_A = 3.5                       # Probe distance cutoff in angstrom.
WATER_O_SELECTION = "name OW OH2"        # Common GROMACS and CHARMM names.
INTERMITTENCY = 0                         # Allowed temporary absence in frames.
COUNT_STATS_MAX_FRAMES = 500              # Occupancy frames; 0 skips counting.
N_PROCS = 4                               # Worker processes.
N_BLOCKS = 5                              # Blocks used for averaging.

CALC_PROTEIN_SHELL = True                 # Water near the whole protein.
CALC_PER_RESIDUE_SHELL = False            # True analyzes every residue by chain.

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
PLOT_X_MAX = 5.0
PLOT_N_BINS = 50
PLOT_BIN_SPACING = 0.5

# Optional custom regions. Leave empty for protein/per-residue water analysis.
TARGET_CONFIGS = []

# Ligand example: replace LIG with the residue name in your topology.
# TARGET_CONFIGS = [{
#     "name": "ligand_shell",
#     "probe_type": "solute",
#     "resnames": ["LIG"],
#     "time_resolution_ns": 0.01,
#     "tau_max_ns": 5.0,
#     "t0_spacing_ns": 0.5,
#     "n_blocks": 5,
# }]
