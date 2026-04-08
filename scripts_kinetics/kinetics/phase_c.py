"""Phase C: aggregate per-t0 SP contributions into the final SP curve."""

import os
import numpy as np

from .io import _sp_origins_path, load_sp_origins_npz


def aggregate_sp_for_region(region, out_dir):
    """Phase C: average per-t0 contributions into the final SP curve.

    Returns (tau_frames, S, counts_tot). If no origin contributed any
    valid samples beyond tau=0, the caller should treat this as an empty
    region (counts_tot[1:].sum() == 0).
    """
    sp_origins_path = _sp_origins_path(out_dir, region.name)
    if not os.path.exists(sp_origins_path):
        raise FileNotFoundError(
            f"No Phase B intermediate for {region.name}: {sp_origins_path}")

    t0_indices, n0_array, sums_2d, counts_2d = load_sp_origins_npz(sp_origins_path)

    sums_tot = sums_2d.sum(axis=0)
    counts_tot = counts_2d.sum(axis=0)

    tau_max_frames = sums_tot.shape[0] - 1
    sp = np.divide(sums_tot, counts_tot,
                   out=np.full_like(sums_tot, np.nan, dtype=np.float64),
                   where=counts_tot > 0)
    if sp.size:
        sp[0] = 1.0
    taus_frames = np.arange(tau_max_frames + 1)
    return taus_frames, sp, counts_tot
