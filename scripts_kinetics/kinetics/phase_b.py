"""Phase B: compute per-t0 SP contributions from contact records.

Loads the contacts intermediate, optionally applies intermittency
correction, then computes one SP curve contribution per t0 origin and
saves the per-t0 array as the sp_origins intermediate.

For large regions (e.g. protein_shell with ~30k contacts/frame), the
contact sets are loaded in blocks to avoid materialising ~40 GB of Python
set objects at once. Each block loads only its window (block origins +
tau_max look-ahead), computes partial SP, then discards the sets before
the next block. Memory is bounded by:
    (block_size + tau_max_frames) × contacts_per_frame × ~28 bytes/int
"""

import os
import time
import numpy as np
from MDAnalysis.lib.correlations import correct_intermittency

from .io import (
    _contacts_path, _sp_origins_path,
    load_contacts_npz, load_contacts_npz_raw, slice_contacts_to_sets,
    save_sp_origins_npz,
)
from .phase_b_worker import compute_origin_sp

# Memory threshold: if raw resindices array > this many bytes, use
# block-wise processing instead of loading all frames into sets at once.
# 500 MB is conservative — leaves plenty of headroom for sets.
_BLOCK_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB

# Default block size (in frames) for block-wise processing.
_DEFAULT_BLOCK_FRAMES = 5000


def compute_sp_for_region(region, out_dir, intermittency, n_procs):
    """Phase B for one region. Loads Phase A intermediate, computes per-t0 SP.

    Automatically chooses between:
    - Full-load mode (small regions): load all contacts into memory as sets
    - Block-wise mode (large regions): load raw arrays, convert to sets in
      blocks to keep memory bounded

    Skips if the sp_origins intermediate already exists.
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

    # Decide mode based on raw data size
    raw_data = np.load(contacts_path)
    raw_size = raw_data["resindices"].nbytes
    n_frames = len(raw_data["frame_indices"])
    del raw_data

    if raw_size <= _BLOCK_THRESHOLD_BYTES:
        return _compute_sp_full_load(
            contacts_path, sp_origins_path, region.name,
            tau_max_frames, t0_step, intermittency, n_frames,
        )
    else:
        return _compute_sp_blockwise(
            contacts_path, sp_origins_path, region.name,
            tau_max_frames, t0_step, intermittency, n_frames,
        )


def _compute_sp_full_load(contacts_path, sp_origins_path, name,
                          tau_max_frames, t0_step, intermittency, n_frames):
    """Small-region path: load all contacts into sets, compute SP at once."""
    _, list_of_sets = load_contacts_npz(contacts_path)

    if intermittency and intermittency > 0:
        list_of_sets = correct_intermittency(list_of_sets,
                                             intermittency=int(intermittency))

    t0_array_indices = list(range(0, n_frames, t0_step))
    n_t0 = len(t0_array_indices)

    print(f"  [Phase B] {name}: {n_t0} origins, tau_max={tau_max_frames} "
          f"frames, full-load, intermittency={intermittency}")

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
    print(f"  [Phase B] {name}: {elapsed:.1f}s ({rate:.1f} t0/s)")

    save_sp_origins_npz(sp_origins_path, t0_array_indices, n0_array,
                        sums_2d, counts_2d)
    return sp_origins_path, elapsed


def _compute_sp_blockwise(contacts_path, sp_origins_path, name,
                          tau_max_frames, t0_step, intermittency, n_frames):
    """Large-region path: process contacts in blocks to bound memory.

    Each block loads frames [block_start, block_start + block_size + tau_max)
    as sets, computes SP for origins in [block_start, block_start + block_size),
    then discards the sets before the next block.
    """
    if intermittency and intermittency > 0:
        print(f"  [Phase B] WARNING: intermittency={intermittency} with "
              f"block-wise processing may miss cross-block gaps. "
              f"Consider intermittency=0 for large regions.")

    # Load raw numpy arrays (compact, stays in memory for all blocks)
    frame_indices, offsets, resindices = load_contacts_npz_raw(contacts_path)
    raw_mb = resindices.nbytes / (1024 * 1024)

    # Determine all t0 indices globally
    all_t0_indices = list(range(0, n_frames, t0_step))
    n_t0 = len(all_t0_indices)
    block_size = _DEFAULT_BLOCK_FRAMES
    n_blocks = max(1, (n_frames + block_size - 1) // block_size)

    print(f"  [Phase B] {name}: {n_t0} origins, tau_max={tau_max_frames} "
          f"frames, block-wise ({n_blocks} blocks of ~{block_size}), "
          f"raw={raw_mb:.0f} MB, intermittency={intermittency}")

    t0_wall = time.perf_counter()
    sums_2d = np.zeros((n_t0, tau_max_frames + 1), dtype=np.float64)
    counts_2d = np.zeros((n_t0, tau_max_frames + 1), dtype=np.int64)
    n0_array = np.zeros(n_t0, dtype=np.int64)

    # Map global t0 indices to their slot in the output arrays
    t0_to_slot = {t0: slot for slot, t0 in enumerate(all_t0_indices)}

    for bi in range(n_blocks):
        block_start = bi * block_size
        block_end = min(block_start + block_size, n_frames)
        # Load extra frames for tau_max look-ahead
        load_end = min(block_end + tau_max_frames, n_frames)

        # Convert this window to sets
        window_sets = slice_contacts_to_sets(offsets, resindices,
                                             block_start, load_end)
        window_len = len(window_sets)

        if intermittency and intermittency > 0:
            window_sets = correct_intermittency(
                window_sets, intermittency=int(intermittency))

        # Find t0 origins that fall in this block
        block_t0s = [t0 for t0 in all_t0_indices
                     if block_start <= t0 < block_end]

        t0_block = time.perf_counter()
        for t0_global in block_t0s:
            t0_local = t0_global - block_start
            # Check look-ahead is sufficient
            if t0_local + tau_max_frames >= window_len:
                # Partial — compute with available range
                effective_tau = window_len - 1 - t0_local
            else:
                effective_tau = tau_max_frames

            n0, sums, counts = compute_origin_sp(
                window_sets, t0_local, effective_tau)

            slot = t0_to_slot[t0_global]
            n0_array[slot] = n0
            # Only fill the tau range we actually computed
            sums_2d[slot, :effective_tau + 1] = sums[:effective_tau + 1]
            counts_2d[slot, :effective_tau + 1] = counts[:effective_tau + 1]

        elapsed_block = time.perf_counter() - t0_block
        print(f"    block {bi + 1}/{n_blocks}: {len(block_t0s)} origins, "
              f"{window_len} frames loaded, {elapsed_block:.1f}s")

        del window_sets  # free memory before next block

    elapsed = time.perf_counter() - t0_wall
    rate = n_t0 / elapsed if elapsed > 0 else 0.0
    print(f"  [Phase B] {name}: {elapsed:.1f}s ({rate:.1f} t0/s, "
          f"{n_blocks} blocks)")

    save_sp_origins_npz(sp_origins_path, all_t0_indices, n0_array,
                        sums_2d, counts_2d)
    return sp_origins_path, elapsed
