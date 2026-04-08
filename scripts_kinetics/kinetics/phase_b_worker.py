"""Phase B worker — compute one t0 origin's SP contribution.

In-process function (no subprocess needed). Phase B's per-t0 work is pure
Python set arithmetic on already-loaded data, so it doesn't suffer from
the fork-after-MDA problem that forces Phase A's worker to be a subprocess.
"""

import numpy as np


def compute_origin_sp(list_of_sets, t0_idx, tau_max):
    """Compute SP contribution for one t0 origin.

    Returns (n0, sum_vec, count_vec) where:
        n0             = |contacts at t0|
        sum_vec[tau]   = |alive_at_tau| / N0   for tau in [0, tau_max]
        count_vec[tau] = 1 if this origin contributed at this tau, else 0

    The sum is the per-origin SP value at each tau; the caller averages
    sum_vec across origins by dividing the per-tau total by counts_tot.
    """
    n = len(list_of_sets)
    initial = list_of_sets[t0_idx]
    n0 = len(initial)

    sums = np.zeros(tau_max + 1, dtype=np.float64)
    counts = np.zeros(tau_max + 1, dtype=np.int64)

    if n0 == 0:
        return 0, sums, counts

    sums[0] = 1.0
    counts[0] = 1
    alive = set(initial)
    inv_n0 = 1.0 / n0
    max_tau = min(tau_max, n - 1 - t0_idx)
    for tau in range(1, max_tau + 1):
        alive &= list_of_sets[t0_idx + tau]
        sums[tau] = len(alive) * inv_n0
        counts[tau] = 1
        # IMPORTANT: do NOT break when alive becomes empty — counts[tau]
        # must stay = 1 for the remaining tau range so the origin-averaged
        # SP is well-defined (matches the old block-parallel code).

    return n0, sums, counts
