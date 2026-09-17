"""TEST 1 - geometry novelty: neighborhood-overlap Jaccard vs conventional
geometries, plus a global/anchor-wise rank-correlation guard.
"""
import numpy as np
from scipy.stats import rankdata, spearmanr

import config
from bootstrap import bootstrap_mean_ci

K = config.K_NEIGHBORS


def _topk_neighbors(dist: np.ndarray, eligible: np.ndarray, k: int):
    n = dist.shape[0]
    d = np.where(eligible, dist, np.inf)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    valid = np.take_along_axis(d, idx, axis=1) < np.inf
    return idx, valid


def jaccard_at_k(dG, dE, eligE, dM, eligM, k=K):
    n = dG.shape[0]
    idxG, validG = _topk_neighbors(dG, np.ones_like(eligE, dtype=bool), k)
    idxE, validE = _topk_neighbors(dE, eligE, k)
    idxM, validM = _topk_neighbors(dM, eligM, k)

    jE = np.full(n, np.nan)
    jM = np.full(n, np.nan)
    for a in range(n):
        ng = set(idxG[a][validG[a]].tolist())
        ne = set(idxE[a][validE[a]].tolist())
        nm = set(idxM[a][validM[a]].tolist())
        if len(ne) > 0:
            jE[a] = len(ng & ne) / k
        if len(nm) > 0:
            jM[a] = len(ng & nm) / k
    return jE, jM


def _row_rank_corr(A: np.ndarray, B: np.ndarray, eligible: np.ndarray):
    n = A.shape[0]
    rhos = np.full(n, np.nan)
    for a in range(n):
        mask = eligible[a].copy()
        mask[a] = False
        if mask.sum() < 10:
            continue
        x = A[a, mask]
        y = B[a, mask]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        rx = rankdata(x)
        ry = rankdata(y)
        rhos[a] = np.corrcoef(rx, ry)[0, 1]
    return rhos


def global_rank_corr(A: np.ndarray, B: np.ndarray, eligible: np.ndarray):
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    mask = eligible[iu]
    x = A[iu][mask]
    y = B[iu][mask]
    rho, _p = spearmanr(x, y)
    return float(rho)


def run_test1(dG, dE, eligE, dM, eligM):
    jE, jM = jaccard_at_k(dG, dE, eligE, dM, eligM, K)

    meanJE, loJE, hiJE = bootstrap_mean_ci(jE, config.N_ANCHOR_BOOTSTRAP, config.SEED_ANCHOR_BOOTSTRAP_TEST1)
    meanJM, loJM, hiJM = bootstrap_mean_ci(jM, config.N_ANCHOR_BOOTSTRAP, config.SEED_ANCHOR_BOOTSTRAP_TEST1 + 1)

    global_rho_E = global_rank_corr(dG, dE, eligE)
    global_rho_M = global_rank_corr(dG, dM, eligM)

    anchor_rho_E = _row_rank_corr(dG, dE, eligE)
    anchor_rho_M = _row_rank_corr(dG, dM, eligM)
    abs_rho_E = np.abs(anchor_rho_E)
    abs_rho_M = np.abs(anchor_rho_M)
    _, _, hi_abs_rho_E = bootstrap_mean_ci(abs_rho_E, config.N_ANCHOR_BOOTSTRAP, config.SEED_ANCHOR_BOOTSTRAP_TEST1 + 2)
    _, _, hi_abs_rho_M = bootstrap_mean_ci(abs_rho_M, config.N_ANCHOR_BOOTSTRAP, config.SEED_ANCHOR_BOOTSTRAP_TEST1 + 3)

    neighborhood_pass = (hiJE < 0.50) and (hiJM < 0.50)
    rank_guard_pass = (
        abs(global_rho_E) < 0.75 and abs(global_rho_M) < 0.75
        and hi_abs_rho_E < 0.75 and hi_abs_rho_M < 0.75
    )
    geometry_novel = neighborhood_pass and rank_guard_pass

    return {
        "geometry_novel": bool(geometry_novel),
        "mean_J20_euclidean": meanJE,
        "mean_J20_euclidean_ci95_lower": loJE,
        "mean_J20_euclidean_ci95_upper": hiJE,
        "mean_J20_mixed": meanJM,
        "mean_J20_mixed_ci95_lower": loJM,
        "mean_J20_mixed_ci95_upper": hiJM,
        "global_rho_geometry_euclidean": global_rho_E,
        "global_rho_geometry_mixed": global_rho_M,
        "mean_abs_anchor_rho_euclidean_ci95_upper": hi_abs_rho_E,
        "mean_abs_anchor_rho_mixed_ci95_upper": hi_abs_rho_M,
    }
