#!/usr/bin/env python3
"""Phase A multi-region worker — process MANY regions in ONE trajectory pass.

Replaces the single-region worker for the common case where multiple
regions share a frame range. Loads the universe once, iterates frames
once, and computes contacts for all regions per frame. Eliminates the
N×re-scan I/O cost when N regions share a chunk.

Usage (always invoked by `phase_a.dispatch_phase_a_multi`):
    python -m kinetics.phase_a_multi_worker \\
        <regions_pickle> <top> <traj> <out_paths_json> <start> <stop>

Args:
    regions_pickle    pickled list[RegionSpec]; all regions must share `stride`
    top, traj         trajectory paths
    out_paths_json    JSON dict {region_name: chunk_npz_path}
    start, stop       trajectory frame range
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance

if __package__ in (None, ""):
    sys.path.insert(0,
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kinetics.io import save_contacts_npz
else:
    from .io import save_contacts_npz


def main():
    if len(sys.argv) != 7:
        print("Usage: phase_a_multi_worker.py <regions_pickle> <top> <traj> "
              "<out_paths_json> <start> <stop>", file=sys.stderr)
        sys.exit(2)

    regions_pickle = sys.argv[1]
    top_path = sys.argv[2]
    traj_path = sys.argv[3]
    out_paths_json = sys.argv[4]
    start = int(sys.argv[5])
    stop = int(sys.argv[6])

    with open(regions_pickle, "rb") as f:
        regions = pickle.load(f)
    with open(out_paths_json) as f:
        out_paths = json.load(f)

    if not regions:
        print("[multi-worker] no regions — nothing to do", flush=True)
        return

    strides = {r.stride for r in regions}
    if len(strides) != 1:
        print(f"[multi-worker] mixed strides {strides} — refusing to multi-pass",
              file=sys.stderr)
        sys.exit(2)
    stride = regions[0].stride

    # Resume-skip: any region whose chunk file already exists with the right
    # frame count can be dropped from this batch.
    n_frames_expected = len(range(start, stop, stride))
    todo = []
    for r in regions:
        cp = out_paths[r.name]
        if os.path.exists(cp):
            try:
                d = np.load(cp)
                if len(d["frame_indices"]) == n_frames_expected:
                    print(f"[multi-worker] {r.name}: chunk already complete — skip",
                          flush=True)
                    continue
            except Exception:
                pass
        todo.append(r)

    if not todo:
        print("[multi-worker] all regions already complete in this chunk",
              flush=True)
        return

    print(f"[multi-worker] loading universe (top={os.path.basename(top_path)}, "
          f"frames=[{start}:{stop}:{stride}], {len(todo)} regions)...",
          flush=True)
    t0 = time.perf_counter()
    # refresh_offsets=False: reuse cached .offsets.npz from prior runs.
    # to_guess=(): skip mass/charge guess overhead.
    u = mda.Universe(top_path, traj_path,
                     refresh_offsets=False, to_guess=())
    t_load = time.perf_counter() - t0
    print(f"[multi-worker] universe in {t_load:.1f}s", flush=True)

    # Per-region: pre-compute static atom indices and mobile atom group + resindices.
    # We do this ONCE per region; then reuse for all frames.
    static_idx_by_region = {}
    mobile_atoms_by_region = {}
    mob_resindices_by_region = {}
    cutoff_by_region = {}
    sets_by_region = {r.name: [] for r in todo}
    skip_region = set()
    for r in todo:
        try:
            sa = u.select_atoms(r.static_sel)
            ma = u.select_atoms(r.mobile_sel)
        except Exception as e:
            print(f"[multi-worker] {r.name}: selection failed ({e}) — skip",
                  flush=True)
            skip_region.add(r.name)
            continue
        if sa.n_atoms == 0 or ma.n_atoms == 0:
            which = "static" if sa.n_atoms == 0 else "mobile"
            print(f"[multi-worker] {r.name}: {which} selection empty — "
                  f"will save empty contacts", flush=True)
            skip_region.add(r.name)
            continue
        static_idx_by_region[r.name] = np.asarray(sa.indices, dtype=np.int64)
        mobile_atoms_by_region[r.name] = ma
        # Direct numpy attribute — replaces O(N) Python list comprehension.
        mob_resindices_by_region[r.name] = np.asarray(ma.resindices,
                                                      dtype=np.int32)
        cutoff_by_region[r.name] = float(r.cutoff_a)

    # Save empty contacts for skipped regions immediately (no per-frame work).
    empty_frames = list(range(start, stop, stride))
    for name in skip_region:
        empty_sets = [set() for _ in empty_frames]
        save_contacts_npz(out_paths[name], empty_frames, empty_sets)

    active = [r for r in todo if r.name not in skip_region]
    if not active:
        print("[multi-worker] all regions empty — done", flush=True)
        return

    # Single trajectory pass.
    print(f"[multi-worker] scanning {n_frames_expected} frames for "
          f"{len(active)} regions (stride={stride})...", flush=True)
    t0 = time.perf_counter()
    frames = []
    sets_per_region = {r.name: [] for r in active}

    all_positions = u.atoms.positions  # initial reference; updated per frame
    for ts in u.trajectory[start:stop:stride]:
        all_positions = ts.positions
        box = ts.dimensions
        for r in active:
            mob = mobile_atoms_by_region[r.name]
            stat_idx = static_idx_by_region[r.name]
            cutoff = cutoff_by_region[r.name]
            mob_resix = mob_resindices_by_region[r.name]
            pairs = capped_distance(
                mob.positions, all_positions[stat_idx],
                cutoff, box=box, return_distances=False,
            )
            if len(pairs) > 0:
                sets_per_region[r.name].append(
                    set(mob_resix[pairs[:, 0]].tolist())
                )
            else:
                sets_per_region[r.name].append(set())
        frames.append(ts.frame)
    t_run = time.perf_counter() - t0
    fps_total = n_frames_expected / t_run if t_run > 0 else 0.0
    fps_per_region = fps_total
    print(f"[multi-worker] {n_frames_expected} frames × {len(active)} regions "
          f"in {t_run:.1f}s ({fps_total:.1f} fps × {len(active)} = "
          f"{fps_total*len(active):.1f} effective fps)", flush=True)

    # Save one npz per region.
    for r in active:
        save_contacts_npz(out_paths[r.name], frames, sets_per_region[r.name])
        print(f"[multi-worker] saved {out_paths[r.name]}", flush=True)


if __name__ == "__main__":
    main()
