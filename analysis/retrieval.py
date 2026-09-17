"""TEST 2 - blind distant-analog retrieval, plus the raw-control
matching-quality gate. Uses only d_E (conventional) to define "far" and to
match controls, and d_G (submitted) to pick the retrieved partner; blind
transport values are used only to score the outcome, never to pick it.
"""
import numpy as np

import config
from bootstrap import bootstrap_mean_ci


def compute_dF(blind_standardized: np.ndarray) -> np.ndarray:
    n, k = blind_standardized.shape
    diff2 = np.zeros((n, n))
    for j in range(k):
        col = blind_standardized[:, j]
        diff2 += (col[:, None] - col[None, :]) ** 2
    return np.sqrt(diff2 / k)


def run_test2(dE: np.ndarray, eligE: np.ndarray, dG: np.ndarray, dF: np.ndarray):
    n = dE.shape[0]
    iu = np.triu_indices(n, k=1)
    e90 = float(np.quantile(dE[iu][eligE[iu]], 0.90))

    dE_work = dE.copy()
    np.fill_diagonal(dE_work, -np.inf)
    far_mask = (dE_work >= e90) & eligE
    np.fill_diagonal(far_mask, False)

    dG_work = np.where(far_mask, dG, np.inf)

    anchors = []
    bG_list = []
    dF_bG = []
    mean_dF_match = []
    mean_abs_gap = []

    for a in range(n):
        candidates = np.flatnonzero(far_mask[a])
        if candidates.size < 21:
            continue
        bG = candidates[np.argmin(dG_work[a, candidates])]
        others = candidates[candidates != bG]
        order = np.argsort(np.abs(dE[a, others] - dE[a, bG]))
        matched = others[order[:20]]

        anchors.append(a)
        bG_list.append(bG)
        dF_bG.append(dF[a, bG])
        mean_dF_match.append(dF[a, matched].mean())
        mean_abs_gap.append(np.abs(dE[a, matched] - dE[a, bG]).mean())

    anchors = np.array(anchors)
    dF_bG = np.array(dF_bG)
    mean_dF_match = np.array(mean_dF_match)
    mean_abs_gap = np.array(mean_abs_gap)
    win = (dF_bG < mean_dF_match).astype(float)
    g_a = mean_abs_gap / e90

    n_eval = len(anchors)

    def ratio_stat(vals):
        return vals["dF_bG"].mean() / vals["mean_dF_match"].mean()

    data = {"dF_bG": dF_bG, "mean_dF_match": mean_dF_match}
    from bootstrap import bootstrap_generic_ci
    R, R_lo, R_hi = bootstrap_generic_ci(data, ratio_stat, config.N_ANCHOR_BOOTSTRAP,
                                          config.SEED_ANCHOR_BOOTSTRAP_TEST2_RETRIEVAL, n_eval)

    W, W_lo, W_hi = bootstrap_mean_ci(win, config.N_ANCHOR_BOOTSTRAP, config.SEED_ANCHOR_BOOTSTRAP_TEST2_RETRIEVAL + 1)
    Gm, Gm_lo, Gm_hi = bootstrap_mean_ci(g_a, config.N_ANCHOR_BOOTSTRAP, config.SEED_ANCHOR_BOOTSTRAP_TEST2_MATCHING)

    matching_pass = (Gm <= 0.10) and (Gm_hi < 0.10)
    retrieval_pass = (R <= 0.95) and (R_hi < 1.0) and (W >= 0.60) and (W_lo > 0.55) and matching_pass

    return {
        "retrieval_useful": bool(retrieval_pass),
        "raw_control_matching_pass": bool(matching_pass),
        "n_evaluable_anchors": int(n_eval),
        "retrieval_ratio": R,
        "retrieval_ratio_ci95_lower": R_lo,
        "retrieval_ratio_ci95_upper": R_hi,
        "retrieval_win_rate": W,
        "retrieval_win_rate_ci95_lower": W_lo,
        "retrieval_win_rate_ci95_upper": W_hi,
        "mean_abs_control_raw_distance_gap": float(mean_abs_gap.mean()),
        "raw_control_matching_gap_ratio": Gm,
        "raw_control_matching_gap_ratio_ci95_lower": Gm_lo,
        "raw_control_matching_gap_ratio_ci95_upper": Gm_hi,
        "raw_far_threshold_q90": e90,
        "_anchors": anchors,
        "_bG": np.array(bG_list),
        "_dF_bG": dF_bG,
        "_mean_dF_match": mean_dF_match,
        "_g_a": g_a,
    }
