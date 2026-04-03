"""Survival probability calculation with block-parallel multiprocessing."""

import os
import time
import warnings
import numpy as np
import MDAnalysis as mda
from multiprocessing import Pool
from MDAnalysis.lib.correlations import correct_intermittency

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from utils import close_universe


# ── Core SP algorithm ────────────────────────────────────────────────────────

def _sp_sums_counts(list_of_sets, tau_max, t0_step=1, t0_stop=None):
    """Continuous survival probability via incremental intersections.

    Returns (sums, counts) arrays for tau = 0..tau_max.
    """
    if tau_max < 0:
        raise ValueError("tau_max must be >= 0")
    if t0_step is None or t0_step < 1:
        raise ValueError("t0_step must be >= 1")

    n = len(list_of_sets)
    if n == 0:
        raise ValueError("list_of_sets must be non-empty")
    if tau_max >= n:
        raise ValueError("tau_max >= length of list_of_sets")

    if t0_stop is None:
        t0_stop = n
    t0_stop = max(0, min(int(t0_stop), n))

    sums = np.zeros(tau_max + 1, dtype=float)
    counts = np.zeros(tau_max + 1, dtype=int)

    for t0 in range(0, t0_stop, t0_step):
        Nt = len(list_of_sets[t0])
        if Nt == 0:
            continue
        sums[0] += 1.0  # SP(0) = 1.0 by definition for each valid origin
        counts[0] += 1
        alive = set(list_of_sets[t0])
        max_tau = min(tau_max, n - 1 - t0)
        for tau in range(1, max_tau + 1):
            alive &= list_of_sets[t0 + tau]
            sums[tau] += len(alive) / float(Nt)
            counts[tau] += 1

    return sums, counts


def _sp_on_block(top, traj, selection, origin_start, origin_stop, load_stop,
                 stride, tau_max_frames, t0_step, intermittency, universe=None):
    """Worker: compute SP for time-origins in [origin_start, origin_stop)."""
    if universe is not None:
        u = universe
    else:
        u = mda.Universe(top, traj, refresh_offsets=True, to_guess=())
    n_total = len(u.trajectory)

    origin_start = max(0, int(origin_start))
    origin_stop = min(int(origin_stop), n_total)
    load_stop = min(int(load_stop), n_total)

    empty = (np.arange(tau_max_frames + 1),
             np.zeros(tau_max_frames + 1),
             np.zeros(tau_max_frames + 1, dtype=int))

    if origin_stop <= origin_start:
        return empty

    n_loaded = len(range(origin_start, load_stop, stride))
    if n_loaded <= tau_max_frames:
        return empty

    sel_atoms = u.select_atoms(selection, updating=True)
    list_of_sets = []
    for _ts in u.trajectory[origin_start:load_stop:stride]:
        list_of_sets.append(set(sel_atoms.residues.resindices))

    if intermittency:
        list_of_sets = correct_intermittency(list_of_sets, intermittency=intermittency)

    n_origins = len(range(origin_start, origin_stop, stride))
    sums, counts = _sp_sums_counts(list_of_sets, tau_max_frames,
                                   t0_step=t0_step, t0_stop=n_origins)
    taus = np.arange(tau_max_frames + 1)
    if universe is None:
        close_universe(u)
    return taus, sums, counts


def _sp_on_block_unpack(args):
    return _sp_on_block(*args)


# ── Block combiner ───────────────────────────────────────────────────────────

def combine_blocks(results):
    """Combine per-block (sums, counts) into global SP timeseries.

    Returns (taus, sp, counts_tot).
    """
    results = [r for r in results if np.any(np.asarray(r[2])[1:] > 0)]
    if not results:
        return np.array([]), np.array([]), np.array([])

    taus_ref = results[0][0]
    sums_tot = np.zeros_like(taus_ref, dtype=float)
    counts_tot = np.zeros_like(taus_ref, dtype=np.int64)

    for taus, sums, counts in results:
        if not np.array_equal(taus, taus_ref):
            raise RuntimeError("Block tau grids differ.")
        sums_tot += np.asarray(sums, float)
        counts_tot += np.asarray(counts, np.int64)

    sp = np.divide(sums_tot, counts_tot,
                   out=np.full_like(sums_tot, np.nan), where=counts_tot > 0)
    if sp.size:
        sp[0] = 1.0
    return taus_ref, sp, counts_tot


# ── Main computation ─────────────────────────────────────────────────────────

def compute_sp(region, top_path, traj_path, start_frame=0, stop_frame=None,
               intermittency=0, count_stats_max_frames=500, n_procs=10,
               universe=None):
    """Compute survival probability for a region.

    Returns dict with tau_ns, S, counts, and metadata.
    If universe is provided, reuses it (avoids MDA XTC re-load crash).
    """
    t0_wall = time.perf_counter()
    print(f"Starting SP calculation for: {region.name}")

    stride = region.stride
    n_blocks = region.n_blocks
    t0_step = region.t0_step
    tau_max_frames = int(region.tau_max_frames)

    print(f"  Settings: resolution={region.time_resolution_ns} ns, "
          f"tau_max={region.tau_max_ns} ns, t0_spacing={region.t0_spacing_ns} ns")
    print(f"  Frames: stride={stride}, tau_max_frames={tau_max_frames}, "
          f"t0_step={t0_step}, n_blocks={n_blocks}")

    # Use provided universe or create new one
    u = universe if universe is not None else mda.Universe(
        top_path, traj_path, refresh_offsets=True, to_guess=())
    nF = len(u.trajectory)
    dt_ps = getattr(u.trajectory, "dt", None)
    if dt_ps is None:
        warnings.warn("Trajectory dt not set; assuming 10.0 ps per frame.")
        dt_ps = 10.0
    dt_ns = float(dt_ps) / 1000.0

    start = max(0, start_frame)
    stop = nF if stop_frame is None else min(stop_frame, nF)

    if tau_max_frames <= 0:
        raise ValueError(f"tau_max_frames must be > 0 for {region.name}")
    n_strided_total = len(range(start, stop, stride))
    if n_strided_total < tau_max_frames + 1:
        raise ValueError(
            f"Frame window too short for tau_max_frames={tau_max_frames} after striding.")

    # Count average molecules in region
    avg_residues = float("nan")
    std_residues = float("nan")
    n_count_frames = 0
    count_step = stride
    count_fps = None

    if count_stats_max_frames == 0:
        print(f"  Skipping count estimation (COUNT_STATS_MAX_FRAMES=0)")
    else:
        if (count_stats_max_frames is not None
                and n_strided_total > count_stats_max_frames):
            factor = int(np.ceil(n_strided_total / float(count_stats_max_frames)))
            count_step = stride * max(1, factor)
        n_count_frames = len(range(start, stop, count_step))
        print(f"  Counting residues over {n_count_frames} frames (step={count_step})...")
        sel_upd = u.select_atoms(region.selection, updating=True)
        counts_list = []
        it = u.trajectory[start:stop:count_step]
        if tqdm is not None:
            it = tqdm(it, total=n_count_frames, desc=f"{region.name} count")
        tc0 = time.perf_counter()
        for _ts in it:
            counts_list.append(sel_upd.residues.n_residues)
        tc = time.perf_counter() - tc0
        if counts_list:
            avg_residues = float(np.mean(counts_list))
            std_residues = float(np.std(counts_list))
            if tc > 0:
                count_fps = len(counts_list) / tc
            print(f"  Count: {avg_residues:.1f} +/- {std_residues:.1f} "
                  f"({count_fps:.2f} frames/s)" if count_fps else
                  f"  Count: {avg_residues:.1f} +/- {std_residues:.1f}")

    # Don't close universe — reuse to avoid MDA XTC re-load crash

    # Early exit: if count is 0 over sampled frames, no contacts exist
    if avg_residues == 0.0 and n_count_frames > 0:
        elapsed = time.perf_counter() - t0_wall
        print(f"  WARNING: No valid data for {region.name} — skipping ({elapsed:.2f}s)")
        return None

    # Partition into blocks (strided-frame space)
    edges_k = np.linspace(0, n_strided_total, n_blocks + 1, dtype=int)
    tasks = []
    for i in range(n_blocks):
        k0, k1 = int(edges_k[i]), int(edges_k[i + 1])
        if k1 <= k0:
            continue
        origin_start = start + k0 * stride
        origin_stop = start + k1 * stride
        load_stop = start + min(k1 + tau_max_frames, n_strided_total) * stride
        tasks.append((top_path, traj_path, region.selection,
                      origin_start, origin_stop, load_stop,
                      stride, tau_max_frames, t0_step, intermittency))

    if not tasks:
        raise ValueError(f"No valid blocks for {region.name}")
    print(f"  Created {len(tasks)} blocks")

    # Multiprocessing
    n_workers = min(n_procs, len(tasks))
    timeout_sec = 600
    if count_fps is not None and count_fps > 0:
        loaded_frames = [len(range(t[3], t[5], t[6])) for t in tasks]
        est_sec = float(np.median(loaded_frames) / count_fps)
        print(f"  ~{int(np.median(loaded_frames))} frames/block, "
              f"~{est_sec / 60.0:.1f} min/block")
        timeout_sec = max(300, est_sec * 5)

    max_retries = 2
    results = []
    failed_blocks = []

    if n_workers <= 1:
        results = [_sp_on_block(*t, universe=u) for t in tasks]
    else:
        pending = list(enumerate(tasks))

        for attempt in range(max_retries + 1):
            if not pending:
                break
            if attempt > 0:
                print(f"  Retry {attempt} for {len(pending)} failed blocks...")
                retry_workers = max(1, n_workers // 2)
            else:
                retry_workers = n_workers

            still_pending = []
            pool = Pool(processes=retry_workers, maxtasksperchild=1)
            try:
                async_results = {
                    idx: pool.apply_async(_sp_on_block_unpack, (task,))
                    for idx, task in pending
                }
                this_timeout = timeout_sec * (attempt + 1)
                deadline = time.perf_counter() + this_timeout
                start_poll = time.perf_counter()
                remaining = dict(async_results)
                completed = []
                last_progress = start_poll

                while remaining and time.perf_counter() < deadline:
                    newly_done = []
                    for idx, ar in remaining.items():
                        if ar.ready():
                            try:
                                res = ar.get(timeout=0)
                                results.append((idx, res))
                                completed.append((idx, time.perf_counter() - start_poll))
                            except Exception as e:
                                print(f"  Block {idx + 1} failed: {e}")
                                still_pending.append((idx, tasks[idx]))
                            newly_done.append(idx)
                    for idx in newly_done:
                        del remaining[idx]
                    if remaining:
                        now = time.perf_counter()
                        if now - last_progress >= 30:
                            elapsed = now - start_poll
                            print(f"  Progress: {len(completed)}/{len(pending)} done, "
                                  f"{elapsed:.0f}s elapsed")
                            last_progress = now
                        time.sleep(1)

                total_elapsed = time.perf_counter() - start_poll
                for idx, elapsed in sorted(completed):
                    print(f"  Block {idx + 1}/{len(tasks)} in {elapsed:.1f}s")
                if completed:
                    print(f"  Total: {len(completed)}/{len(pending)} in {total_elapsed:.1f}s")

                for idx in sorted(remaining.keys()):
                    print(f"  Block {idx + 1} timed out")
                    still_pending.append((idx, tasks[idx]))

            finally:
                pool.terminate()
                pool.join()
                if remaining:
                    print(f"  Killed {len(remaining)} hung worker(s)")

            pending = still_pending

        if pending:
            failed_blocks = [idx for idx, _ in pending]
            print(f"  WARNING: {len(failed_blocks)} blocks failed after retries")

        results = [res for _, res in sorted(results, key=lambda x: x[0])]

    if failed_blocks:
        print(f"  Proceeding with {len(results)}/{len(tasks)} blocks")

    taus_frames, S, counts_tot = combine_blocks(results)
    if S.size == 0:
        elapsed = time.perf_counter() - t0_wall
        print(f"  WARNING: No valid data for {region.name} — skipping ({elapsed:.2f}s)")
        return None

    tau_ns = taus_frames * dt_ns * stride
    elapsed = time.perf_counter() - t0_wall
    print(f"  Completed {region.name} in {elapsed:.2f}s")

    return {
        "tau_ns": tau_ns,
        "S": S,
        "counts": counts_tot,
        "avg_residues": avg_residues,
        "std_residues": std_residues,
        "n_count_frames": n_count_frames,
        "count_step": count_step,
        "dt_ns": dt_ns,
        "start_frame": start,
        "stop_frame": stop,
        "time_taken": elapsed,
        "n_blocks_completed": len(results),
        "n_blocks_total": len(tasks),
    }
