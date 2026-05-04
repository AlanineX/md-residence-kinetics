"""Phase A: parallel contact recording via subprocess.Popen.

Two dispatch modes:

1. ``dispatch_phase_a_subprocess`` — one region at a time, frame range
   split into N chunks. Used for protein_shell and standalone regions.

2. ``dispatch_phase_a_multi`` — ALL regions in one trajectory pass per
   chunk. Massive speedup when chunk size > 1 region; eliminates the
   N×re-scan I/O cost.

In both modes failed chunks are retried up to 3 times and launches are
NOT staggered (the 2-s stagger from the original was a workaround for an
old MDA PDB-parser race that does not affect the current 2.x line).
"""

import json
import os
import pickle
import subprocess
import sys
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

        # Launch all subprocesses immediately. MDA 2.x no longer suffers
        # the PDB-parser race that required the older 2-s stagger.
        running = []  # (popen, ci, f_start, f_stop, chunk_path, log_path, log_f, retry)
        for ci, f_start, f_stop, chunk_path in chunk_specs:
            p, log_path, log_f = _launch_one(ci, f_start, f_stop, chunk_path)
            running.append((p, ci, f_start, f_stop, chunk_path, log_path, log_f, 0))

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


# ── Multi-region single-pass dispatch ──────────────────────────────────────
def dispatch_phase_a_multi(regions, top_path, traj_path, out_dir,
                           start, stop, n_procs, log_dir=None):
    """Process multiple regions in ONE trajectory pass per chunk.

    Trajectory I/O is the dominant cost; with N regions all sharing the
    same frame range, this avoids re-reading the trajectory N times.

    Each chunk subprocess loads the universe once, then iterates frames
    once, computing contacts for ALL regions per frame. Output: one
    chunk-npz per region per chunk (same format as the single-region
    path, so the merge code is unchanged).

    All regions must share the same `stride`. Mixed strides would force
    different frame iterations, defeating the single-pass benefit; if
    you have mixed strides, group same-stride regions and call this
    function once per group.

    Returns {region_name: elapsed_seconds_total}.
    """
    if not regions:
        return {}

    strides = {r.stride for r in regions}
    if len(strides) != 1:
        raise ValueError(
            f"dispatch_phase_a_multi requires uniform stride, got {strides}. "
            "Group regions by stride and call once per group."
        )
    stride = regions[0].stride

    package_dir = os.path.dirname(os.path.abspath(__file__))   # kinetics/
    package_parent = os.path.dirname(package_dir)              # scripts_kinetics/

    pickle_dir = tempfile.mkdtemp(prefix="phase_a_multi_")
    regions_pkl = os.path.join(pickle_dir, "regions.pkl")
    with open(regions_pkl, "wb") as f:
        pickle.dump(list(regions), f)

    if log_dir is None:
        log_dir = os.path.join(out_dir, "_intermediate", "phase_a_logs")
    os.makedirs(log_dir, exist_ok=True)
    chunks_dir = os.path.join(out_dir, "_intermediate", "contacts_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    elapsed_map = {r.name: 0.0 for r in regions}
    t0_global = time.perf_counter()

    # Resume-skip whole regions whose final file already exists & is complete.
    n_expected = len(range(start, stop, stride))
    todo = []
    for r in regions:
        final_path = _contacts_path(out_dir, r.name)
        if os.path.exists(final_path):
            try:
                d = np.load(final_path)
                if len(d["frame_indices"]) == n_expected:
                    print(f"  [skip] {r.name}: contacts already exist")
                    continue
            except Exception:
                pass
        todo.append(r)
    if not todo:
        print("[dispatch-multi] all regions already complete, nothing to do")
        return elapsed_map

    print(f"[dispatch-multi] Phase A on {len(todo)} regions in ONE trajectory "
          f"pass per chunk, split into {n_procs} chunks")

    # Build chunk frame ranges (uniform across all regions).
    all_frames = list(range(start, stop, stride))
    n_total = len(all_frames)
    n_chunks = max(1, min(n_procs, n_total))
    chunk_size = (n_total + n_chunks - 1) // n_chunks

    chunk_specs = []
    for ci in range(n_chunks):
        i0 = ci * chunk_size
        i1 = min(i0 + chunk_size, n_total)
        if i0 >= i1:
            continue
        f_start = all_frames[i0]
        f_stop = all_frames[i1 - 1] + stride  # exclusive
        # Per-region chunk paths AND a JSON file mapping region → chunk path
        out_paths = {
            r.name: os.path.join(chunks_dir, f"{r.name}.part_{ci:03d}.npz")
            for r in todo
        }
        out_paths_json = os.path.join(pickle_dir, f"out_paths_{ci:03d}.json")
        with open(out_paths_json, "w") as f:
            json.dump(out_paths, f)
        chunk_specs.append((ci, f_start, f_stop, out_paths, out_paths_json))

    actual_chunks = len(chunk_specs)
    print(f"  {n_total} frames in {actual_chunks} chunks "
          f"× {len(todo)} regions/chunk")

    t_phase = time.perf_counter()

    def _launch_one(ci, f_start, f_stop, out_paths_json):
        log_path = os.path.join(log_dir, f"_multi.part_{ci:03d}.log")
        log_f = open(log_path, "w")
        cmd = [
            sys.executable, "-u", "-m", "kinetics.phase_a_multi_worker",
            regions_pkl, top_path, traj_path,
            out_paths_json, str(f_start), str(f_stop),
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

    running = []  # (p, ci, f_start, f_stop, out_paths, out_paths_json, log_f, retry)
    for ci, f_start, f_stop, out_paths, out_paths_json in chunk_specs:
        p, log_path, log_f = _launch_one(ci, f_start, f_stop, out_paths_json)
        running.append((p, ci, f_start, f_stop, out_paths, out_paths_json,
                        log_path, log_f, 0))

    any_failed = False
    max_retries = 3
    idx = 0
    while idx < len(running):
        (p, ci, f_start, f_stop, out_paths, out_paths_json,
         log_path, log_f, retry) = running[idx]
        ret = p.wait()
        log_f.close()
        if ret != 0:
            if retry < max_retries:
                print(f"    [retry {retry+1}/{max_retries}] multi chunk {ci}: rc={ret}")
                time.sleep(1.0 + retry)
                p2, log_path2, log_f2 = _launch_one(ci, f_start, f_stop, out_paths_json)
                running[idx] = (p2, ci, f_start, f_stop, out_paths,
                                out_paths_json, log_path2, log_f2, retry + 1)
                continue
            any_failed = True
            print(f"    [FAIL] multi chunk {ci}: rc={ret} after {max_retries} retries")
            try:
                with open(log_path) as fh:
                    for line in fh.readlines()[-12:]:
                        print(f"      | {line.rstrip()}")
            except Exception:
                pass
        idx += 1

    elapsed_chunks = time.perf_counter() - t_phase

    # Merge chunks per region.
    if not any_failed:
        for r in todo:
            t_merge0 = time.perf_counter()
            chunk_paths = [
                os.path.join(chunks_dir, f"{r.name}.part_{ci:03d}.npz")
                for ci, *_ in chunk_specs
            ]
            chunk_paths = [c for c in chunk_paths if os.path.exists(c)]
            if not chunk_paths:
                print(f"  [WARN] {r.name}: no chunks found to merge")
                continue
            final_path = _contacts_path(out_dir, r.name)
            merge_chunk_npz(chunk_paths, final_path)
            for cp in chunk_paths:
                try: os.remove(cp)
                except Exception: pass
            t_merge = time.perf_counter() - t_merge0
            elapsed_map[r.name] = elapsed_chunks + t_merge

    fps = (n_total / elapsed_chunks) if elapsed_chunks > 0 else 0.0
    print(f"  [done] {len(todo)} regions in {elapsed_chunks:.1f}s "
          f"({fps:.1f} fps × {len(todo)} regions = "
          f"{fps*len(todo):.1f} effective fps total)")

    # Cleanup pickle dir.
    try:
        os.remove(regions_pkl)
        for f in os.listdir(pickle_dir):
            os.remove(os.path.join(pickle_dir, f))
        os.rmdir(pickle_dir)
    except Exception:
        pass

    return elapsed_map
