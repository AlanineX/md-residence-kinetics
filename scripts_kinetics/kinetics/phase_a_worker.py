#!/usr/bin/env python3
"""Phase A worker — processes one frame chunk in a fresh Python interpreter.

Run as a subprocess by `phase_a.dispatch_phase_a_subprocess`. This file is
intentionally a standalone script (not just an importable function) because
MDAnalysis cannot be safely re-loaded after a fork — each chunk needs its
own clean process.

Invoke either way:
    python -m kinetics.phase_a_worker <region_pickle> <top> <traj> <out_path> <start> <stop>
    python kinetics/phase_a_worker.py <region_pickle> <top> <traj> <out_path> <start> <stop>

Args:
    region_pickle:  pickled RegionSpec (needs name, static_sel, mobile_sel,
                    cutoff_a, stride)
    top, traj:      paths to topology and trajectory
    out_path:       full path of the .npz output file
    start, stop:    inclusive/exclusive frame indices for this chunk
"""

import os
import sys
import pickle
import time
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance

# When invoked as `python kinetics/phase_a_worker.py ...` (no -m), Python
# does not set up the package context, so a relative import fails. Add the
# parent of this file's directory to sys.path and import absolutely.
if __package__ in (None, ""):
    sys.path.insert(0,
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kinetics.io import save_contacts_npz
else:
    from .io import save_contacts_npz


def main():
    if len(sys.argv) != 7:
        print("Usage: phase_a_worker.py <region_pickle> <top> <traj> "
              "<out_path> <start> <stop>", file=sys.stderr)
        sys.exit(2)

    region_pickle = sys.argv[1]
    top_path = sys.argv[2]
    traj_path = sys.argv[3]
    contacts_path = sys.argv[4]
    start = int(sys.argv[5])
    stop = int(sys.argv[6])

    with open(region_pickle, "rb") as f:
        region = pickle.load(f)

    stride = region.stride
    n_frames_expected = len(range(start, stop, stride))

    # Resume check (defensive — manager already does the same check)
    if os.path.exists(contacts_path):
        try:
            data = np.load(contacts_path)
            if len(data["frame_indices"]) == n_frames_expected:
                print(f"[worker {region.name}] "
                      f"{os.path.basename(contacts_path)}: already exists "
                      f"({n_frames_expected} frames) — skip", flush=True)
                return
        except Exception:
            pass

    print(f"[worker {region.name}] loading universe...", flush=True)
    t0 = time.perf_counter()
    # refresh_offsets=False reuses the cached .offsets.npz that the
    # parent process built; saves ~10-30s per subprocess on a 50001-frame
    # trajectory. to_guess=() skips mass/charge guessing.
    u = mda.Universe(top_path, traj_path,
                     refresh_offsets=False, to_guess=())
    t_load = time.perf_counter() - t0

    static_atoms = u.select_atoms(region.static_sel)
    mobile_atoms = u.select_atoms(region.mobile_sel)
    if static_atoms.n_atoms == 0 or mobile_atoms.n_atoms == 0:
        # Empty selections at the reference frame are valid for dynamic
        # selections (probes may visit later). For a STATIC selection that
        # is genuinely empty, the result is just empty contact sets for
        # every frame — Phase B/C will detect this and skip cleanly.
        # We save an empty intermediate and exit 0 instead of erroring,
        # which would otherwise trigger 3 retries × N chunks of wasted
        # subprocess startups.
        which = "static" if static_atoms.n_atoms == 0 else "mobile"
        print(f"[worker {region.name}] {which} selection empty — saving "
              f"empty contacts for {n_frames_expected} frames", flush=True)
        empty_frames = list(range(start, stop, stride))
        empty_sets = [set() for _ in empty_frames]
        save_contacts_npz(contacts_path, empty_frames, empty_sets)
        return

    # Direct numpy attribute access — replaces the older list comprehension
    # that walked Python-level Atom objects (O(N) Python iteration).
    mob_resix = np.asarray(mobile_atoms.resindices, dtype=np.int32)
    cutoff = float(region.cutoff_a)

    print(f"[worker {region.name}] universe in {t_load:.1f}s, "
          f"processing {n_frames_expected} frames...", flush=True)

    t0 = time.perf_counter()
    frames = []
    sets = []
    for ts in u.trajectory[start:stop:stride]:
        pairs = capped_distance(
            mobile_atoms.positions, static_atoms.positions,
            cutoff, box=ts.dimensions, return_distances=False,
        )
        if len(pairs) > 0:
            sets.append(set(mob_resix[pairs[:, 0]].tolist()))
        else:
            sets.append(set())
        frames.append(ts.frame)
    t_run = time.perf_counter() - t0
    fps = n_frames_expected / t_run if t_run > 0 else 0.0
    print(f"[worker {region.name}] done: {n_frames_expected} frames in "
          f"{t_run:.1f}s ({fps:.2f} fps)", flush=True)

    save_contacts_npz(contacts_path, frames, sets)
    print(f"[worker {region.name}] saved {contacts_path}", flush=True)


if __name__ == "__main__":
    main()
