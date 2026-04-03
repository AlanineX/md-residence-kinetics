# Water/Solute Survival Probability Analysis

Compute survival probability (SP) of water or solute molecules near protein regions from MD trajectories, then fit exponential decay models.

## Usage

```bash
pip install -r requirements.txt

# 1. Edit config_extract.py (paths, targets, settings)
# 2. Run extraction
cd scripts_kinetics
python run_extract.py

# 3. (Optional) Re-plot with different settings
python run_plot.py
```

## Config Reference (`config_extract.py`)

### Trajectory

| Key | Meaning |
|-----|---------|
| `TOP_PATH` | Topology file (PDB/GRO/TPR) |
| `TRAJ_PATH` | Trajectory file (XTC/TRR) |
| `OUT_DIR` | Output directory for results |
| `START_FRAME` / `STOP_FRAME` | Frame range to analyze (`None` = full trajectory) |

### Analysis Parameters

| Key | Default | Meaning |
|-----|---------|---------|
| `FIRST_SHELL_A` | 3.5 | Distance cutoff in angstrom for "near protein" selection |
| `WATER_O_SELECTION` | `"name OW OH2"` | MDAnalysis selection for water oxygens (OW for GROMACS, OH2 for CHARMM) |
| `INTERMITTENCY` | 0 | Allow molecule to leave for N frames and still count as "survived" |
| `COUNT_STATS_MAX_FRAMES` | 500 | Max frames for counting average occupancy (0 = skip) |
| `N_PROCS` | 10 | Worker processes for multiprocessing |
| `N_BLOCKS` | 1 | Split trajectory into N blocks (for block averaging / error estimation) |
| `DO_EXP_FIT` | True | Fit single- and bi-exponential decay after extraction |

### SP Timing Settings

Used in `PROTEIN_SHELL_SETTINGS`, `MDA_SETTINGS`, `PER_RESIDUE_SETTINGS`, etc.:

| Key | Example | Meaning |
|-----|---------|---------|
| `time_resolution_ns` | 0.01 | Time step between frames used in SP calculation. Frames are strided so that each step equals this value. Lower = finer resolution but slower. |
| `tau_max_ns` | 5.0 | Maximum lag time (tau) for the SP curve. **Must be < trajectory_length / 2**, ideally < trajectory_length / 3. |
| `t0_spacing_ns` | 0.5 | Spacing between time origins. Smaller = more origins = better statistics but slower. Aim for >= 20 origins. |
| `n_blocks` | 1 | Number of trajectory blocks for this target. Each block computes SP independently, then results are combined. |

### Protein Shell & Per-Residue

| Key | Values | Meaning |
|-----|--------|---------|
| `CALC_PROTEIN_SHELL` | `True`/`False` | Compute SP for all water near protein |
| `CALC_PER_RESIDUE_SHELL` | `None`/`False` = skip, `True` = all residues, `"0:49"` = position range | Per-residue SP (chain-aware: equivalent residues across chains are combined) |

### Target Configs

`TARGET_CONFIGS` is a list of dicts, each defining a region:

```python
{"name": "pocket_water",
 "probe_type": "water",           # "water" or "solute"
 "resnames": ["MDA"],             # for solute probe_type
 "resid_ranges": ["52 53 87"],    # protein residues defining the region
 "chains": "ABCD",                # chain filter (optional)
 "time_resolution_ns": 0.01,
 "tau_max_ns": 5.0,
 "t0_spacing_ns": 0.5,
 "n_blocks": 1}
```

## Output

- `sp_<name>.csv` — tau (ns) vs SP curve
- `summary_<name>.txt` — fit parameters, residence times
- `plots/` — SP decay plots with fits
- `run_settings.log` — full run configuration

## Known Issue

MDAnalysis <= 2.9.0 has a memory corruption bug when re-loading large XTC files (>500K atoms) in the same process. This is worked around by loading the Universe once and reusing it throughout the pipeline.
