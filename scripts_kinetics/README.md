# md-residence-kinetics

Survival probability (SP) analysis for water and solute residence kinetics from MD trajectories. Computes SP decay curves, fits single/bi-exponential models, and provides model-free metrics (RMST).

## Installation

```bash
git clone https://github.com/AlanineX/md-residence-kinetics.git
cd md-residence-kinetics/scripts_kinetics
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Copy and edit the example config
cp config_extract_example.py config_extract.py
# Edit config_extract.py: set TOP_PATH, TRAJ_PATH, OUT_DIR, and targets

# 2. Run extraction + fitting
python run_extract.py

# 3. (Optional) Re-fit/re-plot existing SP CSVs with different settings
python run_plot.py
```

## What It Does

1. **Counts** average probe (water/solute) occupancy per region
2. **Computes** survival probability S(tau) via block-parallel time correlation
3. **Fits** single-exp `(1-c)*exp(-t/tau) + c` and bi-exp models
4. **Calculates** model-free metrics: RMST(t\*) and S(t\*) at fixed horizons
5. **Outputs** per-region CSVs, summary text files, SVG plots, and aggregate fitting CSVs

## Config Reference (`config_extract.py`)

### Trajectory

| Key | Meaning |
|-----|---------|
| `TOP_PATH` | Topology file (PDB/GRO/TPR) |
| `TRAJ_PATH` | Trajectory file (XTC/TRR) |
| `OUT_DIR` | Output directory |
| `START_FRAME` / `STOP_FRAME` | Frame range (`None` = full trajectory) |

### Analysis

| Key | Default | Meaning |
|-----|---------|---------|
| `FIRST_SHELL_A` | 3.5 | Distance cutoff (angstrom) for probe selection |
| `WATER_O_SELECTION` | `"name OW OH2"` | MDAnalysis selection for water oxygens |
| `INTERMITTENCY` | 0 | Allow probe to leave for N frames and still count as survived |
| `COUNT_STATS_MAX_FRAMES` | 500 | Max frames for occupancy estimate (0 = skip) |
| `N_PROCS` | 10 | Worker processes for multiprocessing |
| `N_BLOCKS` | 1 | Trajectory blocks for block averaging |
| `DO_EXP_FIT` | True | Fit exponential decay models |

### SP Timing Settings

Used in `PROTEIN_SHELL_SETTINGS`, `PER_RESIDUE_SETTINGS`, and per-target configs:

| Key | Example | Meaning |
|-----|---------|---------|
| `time_resolution_ns` | 0.01 | Time step for SP calculation. Frames are strided to match. |
| `tau_max_ns` | 5.0 | Maximum lag time. **Must be < trajectory_length / 2.** |
| `t0_spacing_ns` | 0.5 | Spacing between time origins. Smaller = better statistics, slower. Aim for >= 20 origins. |
| `n_blocks` | 1 | Blocks for this target (overrides global `N_BLOCKS`). |

### Protein Shell & Per-Residue

| Key | Values | Meaning |
|-----|--------|---------|
| `CALC_PROTEIN_SHELL` | `True`/`False` | SP for all water near protein |
| `CALC_PER_RESIDUE_SHELL` | `None` = skip, `True` = all, `"0:49"` = range | Per-residue SP (chain-aware: equivalent residues across chains are combined) |

### Target Configs

`TARGET_CONFIGS` is a list of dicts defining custom regions:

```python
TARGET_CONFIGS = [
    {"name": "my_region",
     "probe_type": "water",           # "water" or "solute"
     "resnames": ["ADP"],             # required for "solute"
     "resid_ranges": ["52 53 87"],    # protein residues (one string per chain)
     "chains": "ABCD",               # optional chain filter
     "time_resolution_ns": 0.01,
     "tau_max_ns": 5.0,
     "t0_spacing_ns": 0.5,
     "n_blocks": 5,
     "description": "Water near my binding site"},
]
```

## Output

| File | Content |
|------|---------|
| `sp_<name>.csv` | tau (ns) vs SP curve |
| `summary_<name>.txt` | Fit parameters, residence times, counts |
| `plots/<name>.svg` | SP decay plot with fitted curves |
| `single_exp_fitting_results.csv` | All regions: 1-exp fit + counts + RMST |
| `bi_exp_fitting_results.csv` | All regions: 2-exp fit + counts + RMST |
| `run_settings.log` | Full run configuration |

### Aggregate CSV Columns

| Column | Meaning |
|--------|---------|
| `avg_count` / `std_count` | Mean probe occupancy in region |
| `tau`, `tau1`, `tau2` | Exponential decay time constants (ns) |
| `c` | Constant offset (non-exchanging fraction) |
| `apparent_res_time` | Integral of (S(t) - c) from 0 to tau_max |
| `fitted_res_time` | alpha1\*tau1 + alpha2\*tau2 (analytical integral to infinity) |
| `t_half_overall` | Time when S(t) drops to midpoint between 1 and c |
| `RMST_1ns`, `RMST_2ns`, `RMST_5ns` | Restricted mean survival time at 1/2/5 ns horizons (model-free) |
| `S_1ns`, `S_2ns`, `S_5ns` | Raw SP value at 1/2/5 ns (model-free) |

## Known Issues

- MDAnalysis <= 2.9.0 has a memory corruption bug when re-loading large XTC files (>500K atoms). Workaround: the pipeline loads the Universe once and reuses it. For subprocess-based runs, each site gets a fresh process.
