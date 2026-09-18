"""TEST 1 -- geometry novelty (Strong Gate 1)."""
import numpy as np
from scipy import stats as sstats

from .bootstrap import bootstrap_ci_indices


def nn20_sets(D, orig_positions, k=20):
    n = D.shape[0]
    neighbor_sets = []
    for i in range(n):
        d = D[i].copy()
        d[i] = np.inf
        order = np.lexsort((orig_positions, d))
        top = order[:k]
        neighbor_sets.append(set(orig_positions[top].tolist()))
    return neighbor_sets


def jaccard_overlap(setsA, setsB, k=20):
    n = len(setsA)
    j = np.empty(n, dtype=float)
    for i in range(n):
        j[i] = len(setsA[i] & setsB[i]) / k
    return j


def upper_tri_pairs(D):
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = D[iu]
    finite = np.isfinite(vals)
    return vals[finite], iu, finite


def run_test1(D_G, D_E, D_M, orig_positions):
    N_G = nn20_sets(D_G, orig_positions)
    N_E = nn20_sets(D_E, orig_positions)
    N_M = nn20_sets(D_M, orig_positions)

    J20_E = jaccard_overlap(N_G, N_E)
    J20_M = jaccard_overlap(N_G, N_M)

    ge_vals, iu, finite = upper_tri_pairs(D_G)
    e_vals = D_E[iu][finite]
    m_vals = D_M[iu][finite]
    global_rho_E, _ = sstats.spearmanr(ge_vals, e_vals)
    finite_m = np.isfinite(D_M[iu])
    global_rho_M, _ = sstats.spearmanr(D_G[iu][finite_m], D_M[iu][finite_m])

    def stat_mean_j20e(idx):
        return np.mean(J20_E[idx])

    def stat_mean_j20m(idx):
        return np.mean(J20_M[idx])

    n = len(J20_E)
    point_e, lo_e, hi_e, _ = bootstrap_ci_indices(n, seed=101002, n_boot=3000, compute_stat_fn=stat_mean_j20e)
    point_m, lo_m, hi_m, _ = bootstrap_ci_indices(n, seed=101002, n_boot=3000, compute_stat_fn=stat_mean_j20m)

    return {
        "J20_E": J20_E,
        "J20_M": J20_M,
        "mean_J20_euclidean": point_e,
        "mean_J20_euclidean_ci95_lower": lo_e,
        "mean_J20_euclidean_ci95_upper": hi_e,
        "mean_J20_mixed": point_m,
        "mean_J20_mixed_ci95_lower": lo_m,
        "mean_J20_mixed_ci95_upper": hi_m,
        "global_rho_geometry_euclidean": global_rho_E,
        "global_rho_geometry_mixed": global_rho_M,
        "gate1_pass": bool(hi_e < 0.33),
    }
