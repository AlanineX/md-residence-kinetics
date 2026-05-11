"""Lightweight region descriptor shared by Method 1 and Method 2.

Originally lived in `scripts_kinetics/utils.py`, but `utils.py` imports
MDAnalysis at module load — too heavy for Method 2 (which only walks
contact NPZ files) and for tests (which want to construct synthetic
regions without a trajectory). Putting the dataclass here keeps it
MDA-free and importable from anywhere in the package.

`utils.py` re-exports `RegionSpec` from this module for backwards compat.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegionSpec:
    name: str
    selection: str
    csv_path: str
    txt_path: str
    description: str = ""
    # Decomposed selection for the contact pipeline (capped_distance based)
    static_sel: str = ""    # protein region of interest
    mobile_sel: str = ""    # the molecule whose residence is tracked (water/solute)
    cutoff_a: float = 3.5
    # User settings (ns)
    time_resolution_ns: float = 0.01
    tau_max_ns: float = 25.0
    t0_spacing_ns: float = 0.5
    # Computed (set by validate_and_compute_settings)
    stride: int = 1
    tau_max_frames: int = 0
    t0_step: int = 1
    actual_resolution_ns: float = 0.0
    n_origins_estimated: int = 0
    valid_origin_range_ns: float = 0.0
