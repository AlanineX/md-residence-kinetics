"""On-disk intermediate IO for the kinetics pipeline.

Contact and SP-origin files use a flat npz layout so saving/loading is
fast and the layout is self-describing for inspection.
"""

import os
import numpy as np


# ── Path helpers ────────────────────────────────────────────────────────────

def _contacts_path(out_dir, region_name):
    return os.path.join(out_dir, "_intermediate", "contacts", f"{region_name}.npz")


def _sp_origins_path(out_dir, region_name):
    return os.path.join(out_dir, "_intermediate", "sp_origins", f"{region_name}.npz")


# ── Contacts npz ────────────────────────────────────────────────────────────

def save_contacts_npz(path, frame_indices, list_of_sets):
    """Save per-frame contact sets as a flat npz with offsets.

    Layout:
        frame_indices: int32[n_frames]    — which frame each entry is for
        offsets:       int64[n_frames+1]  — start/end into resindices
        resindices:    int32[total]       — concatenated set contents
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(list_of_sets)
    sizes = np.fromiter((len(s) for s in list_of_sets), dtype=np.int64, count=n)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    total = int(offsets[-1])
    flat = np.empty(total, dtype=np.int32)
    pos = 0
    for s in list_of_sets:
        if s:
            flat[pos:pos + len(s)] = np.fromiter(s, dtype=np.int32, count=len(s))
            pos += len(s)
    # np.savez appends '.npz' to the filename if missing — write to a temp
    # file with explicit .npz then rename atomically.
    tmp = path + ".tmp.npz"
    np.savez(tmp,
             frame_indices=np.asarray(frame_indices, dtype=np.int32),
             offsets=offsets,
             resindices=flat)
    os.replace(tmp, path)


def load_contacts_npz(path):
    """Load contact records back as (frame_indices, list_of_sets)."""
    data = np.load(path)
    frame_indices = data["frame_indices"]
    offsets = data["offsets"]
    flat = data["resindices"]
    list_of_sets = []
    for i in range(len(frame_indices)):
        a, b = int(offsets[i]), int(offsets[i + 1])
        list_of_sets.append(set(flat[a:b].tolist()))
    return frame_indices, list_of_sets


# ── SP origins npz ──────────────────────────────────────────────────────────

def save_sp_origins_npz(path, t0_indices, n0_array, sums_2d, counts_2d):
    """Save per-t0 SP contributions.

    Layout:
        t0_indices: int64[n_t0]           — origin frame indices
        n0:         int64[n_t0]           — initial set size at each t0
        sums:       float64[n_t0, tau+1]  — per-origin sum_{tau} = |alive|/N0
        counts:     int64[n_t0, tau+1]    — 1 where the origin contributed,
                                            0 where it ran past trajectory end
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp,
             t0_indices=np.asarray(t0_indices, dtype=np.int64),
             n0=np.asarray(n0_array, dtype=np.int64),
             sums=np.asarray(sums_2d, dtype=np.float64),
             counts=np.asarray(counts_2d, dtype=np.int64))
    os.replace(tmp, path)


def load_sp_origins_npz(path):
    data = np.load(path)
    return (data["t0_indices"], data["n0"],
            data["sums"], data["counts"])


# ── Chunk merge (Phase A) ───────────────────────────────────────────────────

def merge_chunk_npz(chunk_paths, final_path):
    """Merge multiple per-chunk contacts npz files into one final npz.

    Each chunk has its own (frame_indices, offsets, resindices). After merge,
    the result is sorted by frame index. Chunks are non-overlapping in our
    Phase A design — each subprocess processes a disjoint frame range.
    """
    all_frames = []
    all_sizes = []
    all_resids = []
    for cp in chunk_paths:
        d = np.load(cp)
        frame_indices = d["frame_indices"]
        offsets = d["offsets"]
        resindices = d["resindices"]
        for i, fi in enumerate(frame_indices):
            a, b = int(offsets[i]), int(offsets[i + 1])
            all_frames.append(int(fi))
            all_sizes.append(b - a)
            all_resids.append(resindices[a:b])

    order = sorted(range(len(all_frames)), key=lambda i: all_frames[i])
    sorted_frames = np.array([all_frames[i] for i in order], dtype=np.int32)
    sorted_sizes = np.array([all_sizes[i] for i in order], dtype=np.int64)
    offsets = np.zeros(len(sorted_sizes) + 1, dtype=np.int64)
    np.cumsum(sorted_sizes, out=offsets[1:])
    total = int(offsets[-1])
    flat = np.empty(total, dtype=np.int32)
    pos = 0
    for i in order:
        n = all_sizes[i]
        flat[pos:pos + n] = all_resids[i].astype(np.int32, copy=False)
        pos += n

    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp = final_path + ".tmp.npz"
    np.savez(tmp,
             frame_indices=sorted_frames,
             offsets=offsets,
             resindices=flat)
    os.replace(tmp, final_path)
