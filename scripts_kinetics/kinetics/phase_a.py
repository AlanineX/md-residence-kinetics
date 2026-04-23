"""Phase A: parallel contact recording via subprocess.Popen.

Each region is processed as a sequential job; within a region the frame
range is split into N chunks that run as fresh Python subprocesses (one
per chunk). Subprocess isolation avoids the fork-after-MDA deadlocks we
hit with multiprocessing.Pool. Failed chunks are retried up to 3 times,
and launches are staggered to avoid simultaneous PDB-parser races.
"""

import os
import sys
import pickle
import subprocess
import tempfile
import time
import numpy as np

from .io import _contacts_path, merge_chunk_npz


def dispatch_phase_a_subprocess(regions, top_path, traj_path, out_dir,
                                start, stop, n_procs, log_dir=None):
    """Run Phase A for a list of regions, sequentially with intra-region
    parallelism.

    For each region:
      1. Split frames [start, stop) by stride into n_procs roughly-equal chunks
      2. Spawn n_procs subprocesses (one per chunk), wait for all
      3. Merge per-chunk npz files into one per-region contacts npz
      4. Move to next region

    Returns {region_name: elapsed_seconds}.
    """
    if not regions:
        return {}

    # phase_a_worker.py lives inside this package. We launch it as a module
    # (`python -m kinetics.phase_a_worker`) so it gets the package context
    # and can use relative imports. The PYTHONPATH must include the parent
    # of the kinetics package — set below in _launch_one.
    package_dir = os.path.dirname(os.path.abspath(__file__))   # kinetics/
    package_parent = os.path.dirname(package_dir)              # scripts_kinetics/
    worker_script = os.path.join(package_dir, "phase_a_worker.py")
    if not os.path.exists(worker_script):
        raise FileNotFoundError(f"phase_a_worker.py not found at {worker_script}")

    pickle_dir = tempfile.mkdtemp(prefix="phase_a_regions_")
    region_pickles = {}
    for r in regions:
        path = os.path.join(pickle_dir, f"{r.name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(r, f)
        region_pickles[r.name] = path

    if log_dir is None:
        log_dir = os.path.join(out_dir, "_intermediate", "phase_a_logs")
    os.makedirs(log_dir, exist_ok=True)
    # chunks_dir is shared by all parallel array tasks writing to the same
    # out_dir. Each region's chunks have unique filenames (region.part_NNN),
    # so no write collision. But we must NOT rmdir this shared directory —
    # another task may be writing to it concurrently.
    chunks_dir = os.path.join(out_dir, "_intermediate", "contacts_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    elapsed_map = {}
    t0_global = time.perf_counter()

    print(f"[dispatch] Phase A on {len(regions)} regions, "
          f"each split into {n_procs} chunks")

    for r in regions:
        # Skip if final file already exists with the right frame count
        final_path = _contacts_path(out_dir, r.name)
        n_expected = len(range(start, stop, r.stride))
        if os.path.exists(final_path):
            try:
                d = np.load(final_path)
                if len(d["frame_indices"]) == n_expected:
                    print(f"  [skip] {r.name}: contacts already exist")
                    elapsed_map[r.name] = 0.0
                    continue
            except Exception:
                pass

        # Build chunk frame ranges
        all_frames = list(range(start, stop, r.stride))
        n_total = len(all_frames)
        n_chunks = max(1, min(n_procs, n_total))
        chunk_size = (n_total + n_chunks - 1) // n_chunks  # ceil

        chunk_specs = []
        for ci in range(n_chunks):
            i0 = ci * chunk_size
            i1 = min(i0 + chunk_size, n_total)
            if i0 >= i1:
                continue
            f_start = all_frames[i0]
            f_stop = all_frames[i1 - 1] + r.stride  # exclusive
            chunk_path = os.path.join(chunks_dir,
                                      f"{r.name}.part_{ci:03d}.npz")
            chunk_specs.append((ci, f_start, f_stop, chunk_path))

        actual_chunks = len(chunk_specs)
        print(f"  [region {r.name}] {n_total} frames in {actual_chunks} chunks, "
              f"spawning {actual_chunks} subprocesses...")

        t_region = time.perf_counter()

        def _launch_one(ci, f_start, f_stop, chunk_path):
            log_path = os.path.join(log_dir, f"{r.name}.part_{ci:03d}.log")
            log_f = open(log_path, "w")
            # Launch as a module so the worker gets full package context.
            # PYTHONPATH must include scripts_kinetics/ for `kinetics` to
            # be importable from the subprocess (it runs in cwd=/tmp).
            cmd = [
                sys.executable, "-u", "-m", "kinetics.phase_a_worker",
                region_pickles[r.name], top_path, traj_path,
                chunk_path, str(f_start), str(f_stop),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = (package_parent
                                 + (os.pathsep + env["PYTHONPATH"]
                                    if env.get("PYTHONPATH") else ""))
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,
                                 cwd="/tmp", start_new_session=True,
                                 env=env)
            return p, log_path, log_f

        # Stagger 2s — MDA's PDB parser appears to need sole access during
        # the first ~1s of import. Without this, simultaneous loads from
        # multiple subprocesses corrupt shared state and crash with bizarre
        # interpreter errors.
        running = []  # (popen, ci, f_start, f_stop, chunk_path, log_path, log_f, retry)
        for ci, f_start, f_stop, chunk_path in chunk_specs:
            p, log_path, log_f = _launch_one(ci, f_start, f_stop, chunk_path)
            running.append((p, ci, f_start, f_stop, chunk_path, log_path, log_f, 0))
            time.sleep(2.0)

        # Wait for all, retrying any that fail (up to 3 attempts)
        chunk_paths = []
        any_failed = False
        max_retries = 3
        idx = 0
        while idx < len(running):
            (p, ci, f_start, f_stop, chunk_path,
             log_path, log_f, retry) = running[idx]
            ret = p.wait()
            log_f.close()
            if ret != 0:
                if retry < max_retries:
                    print(f"    [retry {retry+1}/{max_retries}] {r.name} chunk {ci}: rc={ret}")
                    time.sleep(1.0 + retry)
                    p2, log_path2, log_f2 = _launch_one(ci, f_start, f_stop, chunk_path)
                    running[idx] = (p2, ci, f_start, f_stop, chunk_path,
                                     log_path2, log_f2, retry + 1)
                    continue  # re-poll same idx
                else:
                    any_failed = True
                    print(f"    [FAIL] {r.name} chunk {ci}: rc={ret} after "
                          f"{max_retries} retries (see {log_path})")
                    try:
                        with open(log_path) as f:
                            for line in f.readlines()[-8:]:
                                print(f"      | {line.rstrip()}")
                    except Exception:
                        pass
            else:
                chunk_paths.append(chunk_path)
            idx += 1

        elapsed = time.perf_counter() - t_region
        elapsed_map[r.name] = elapsed
        if any_failed:
            print(f"  [FAIL] {r.name}: some chunks failed, leaving partial files")
            continue

        # Merge chunks into final per-region file
        t_merge = time.perf_counter()
        merge_chunk_npz(chunk_paths, final_path)
        merge_elapsed = time.perf_counter() - t_merge

        # Clean up per-chunk files (now merged)
        for cp in chunk_paths:
            try:
                os.remove(cp)
            except Exception:
                pass

        fps = n_total / elapsed if elapsed > 0 else 0.0
        fps_per_core = fps / actual_chunks
        print(f"  [done]   {r.name}: {elapsed:.1f}s "
              f"({fps:.1f} fps total, {fps_per_core:.2f} fps/core, "
              f"merge {merge_elapsed:.2f}s)")

    total = time.perf_counter() - t0_global
    print(f"[dispatch] Phase A total: {total:.1f}s wall")

    # Cleanup pickles (per-task tmpdir — safe to remove)
    for path in region_pickles.values():
        try:
            os.remove(path)
        except Exception:
            pass
    try:
        os.rmdir(pickle_dir)
    except Exception:
        pass
    # NOTE: do NOT rmdir chunks_dir — it's shared across parallel array
    # tasks. Another task may be writing chunk files to it right now.
    # The directory is harmless to leave around (just empty after merge).

    return elapsed_map
