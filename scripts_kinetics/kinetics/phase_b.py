"""Phase B: orchestrate per-t0 SP computation for one region.

Loads the contacts intermediate, optionally applies intermittency
correction, then runs `compute_origin_sp` (from phase_b_worker) for every
t0 origin and saves the per-t0 array as the sp_origins intermediate.
"""

import os
import time
import numpy as np
from MDAnalysis.lib.correlations import correct_intermittency

from .io import (
    _contacts_path, _sp_origins_path,
    load_contacts_npz, save_sp_origins_npz,
)
from .phase_b_worker import compute_origin_sp


def compute_sp_for_region(region, out_dir, intermittency, n_procs):
    """Phase B for one region. Loads Phase A intermediate, computes per-t0 SP.

    Skips if the sp_origins intermediate already exists. The `n_procs`
    argument is currently unused (Phase B is single-process — fast enough
    that the cross-worker pickling cost would dominate); it is kept in the
    signature for symmetry with Phase A.
    """
    contacts_path = _contacts_path(out_dir, region.name)
    sp_origins_path = _sp_origins_path(out_dir, region.name)
    tau_max_frames = int(region.tau_max_frames)
    t0_step = int(region.t0_step)

    if os.path.exists(sp_origins_path):
        try:
            np.load(sp_origins_path)
            print(f"  [Phase B skip] {region.name}: sp_origins already exist")
            return sp_origins_path, 0.0
        except Exception:
            pass  # corrupt file, re-build

    if not os.path.exists(contacts_path):
        raise FileNotFoundError(
            f"No Phase A intermediate for {region.name}: {contacts_path}")

    frame_indices, list_of_sets = load_contacts_npz(contacts_path)
    n_frames = len(list_of_sets)

    # Optional intermittency correction (matches the old code path).
    if intermittency and intermittency > 0:
        list_of_sets = correct_intermittency(list_of_sets,
                                             intermittency=int(intermittency))

    # Subsample t0 indices in array index space (post-stride)
    t0_array_indices = list(range(0, n_frames, t0_step))
    n_t0 = len(t0_array_indices)

    print(f"  [Phase B] {region.name}: {n_t0} origins, tau_max={tau_max_frames} "
          f"frames, intermittency={intermittency}")

    t0 = time.perf_counter()
    sums_2d = np.zeros((n_t0, tau_max_frames + 1), dtype=np.float64)
    counts_2d = np.zeros((n_t0, tau_max_frames + 1), dtype=np.int64)
    n0_array = np.zeros(n_t0, dtype=np.int64)

    for slot, t0_idx in enumerate(t0_array_indices):
        n0, sums, counts = compute_origin_sp(list_of_sets, t0_idx, tau_max_frames)
        sums_2d[slot] = sums
        counts_2d[slot] = counts
        n0_array[slot] = n0

    elapsed = time.perf_counter() - t0
    rate = n_t0 / elapsed if elapsed > 0 else 0.0
    print(f"  [Phase B] {region.name}: {elapsed:.1f}s ({rate:.1f} t0/s)")

    save_sp_origins_npz(sp_origins_path, t0_array_indices, n0_array,
                        sums_2d, counts_2d)
    return sp_origins_path, elapsed
