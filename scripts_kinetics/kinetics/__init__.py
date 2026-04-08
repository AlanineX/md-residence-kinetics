"""Survival probability pipeline (kinetics package).

Three-phase pipeline with on-disk intermediates:

  Phase A — Contact recording
      Read trajectory once per region, build per-frame contact records.
      Parallelised across frame chunks via subprocess.Popen on a fresh
      Python interpreter per chunk (avoids the fork-after-MDA deadlocks
      we hit with multiprocessing.Pool).
      → `_intermediate/contacts/{region}.npz`

  Phase B — SP from contacts
      Load contact records, optionally apply intermittency correction,
      compute per-t0 SP contributions.
      → `_intermediate/sp_origins/{region}.npz`

  Phase C — Aggregate
      Sum per-t0 contributions into the final SP curve.

Public API (imported here for convenience):
    dispatch_phase_a_subprocess  — Phase A driver
    compute_sp_for_region        — Phase B driver
    aggregate_sp_for_region      — Phase C
    save_contacts_npz / load_contacts_npz
    save_sp_origins_npz / load_sp_origins_npz
    _contacts_path / _sp_origins_path
"""

from .io import (
    save_contacts_npz,
    load_contacts_npz,
    save_sp_origins_npz,
    load_sp_origins_npz,
    _contacts_path,
    _sp_origins_path,
)
from .phase_a import dispatch_phase_a_subprocess
from .phase_b import compute_sp_for_region
from .phase_c import aggregate_sp_for_region

__all__ = [
    "dispatch_phase_a_subprocess",
    "compute_sp_for_region",
    "aggregate_sp_for_region",
    "save_contacts_npz",
    "load_contacts_npz",
    "save_sp_origins_npz",
    "load_sp_origins_npz",
    "_contacts_path",
    "_sp_origins_path",
]
