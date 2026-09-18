"""TEST 2 -- blind distant-analog retrieval (Strong Gates 2 & 3, raw-matching validity)."""
import numpy as np

from .bootstrap import bootstrap_ci_indices


def compute_e090(D_E):
    n = D_E.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = D_E[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, 0.90))


def run_test2(D_G, D_E, D_F, orig_positions):
    n = D_G.shape[0]
    e090 = compute_e090(D_E)

    retrieved_dF = np.full(n, np.nan)
    matched_dF_mean = np.full(n, np.nan)
    matched_gap = np.full(n, np.nan)
    bG_of = np.full(n, -1, dtype=int)
    evaluable = np.zeros(n, dtype=bool)

    for a in range(n):
        d_e_row = D_E[a, :]
        mask = np.isfinite(d_e_row) & (d_e_row >= e090)
        mask[a] = False
        C_idx = np.where(mask)[0]
        if len(C_idx) < 1:
            continue
        d_g_vals = D_G[a, C_idx]
        order = np.lexsort((orig_positions[C_idx], d_g_vals))
        bG_local = C_idx[order[0]]

        remaining = C_idx[C_idx != bG_local]
        if len(remaining) < 20:
            continue
        target_dE = D_E[a, bG_local]
        gaps = np.abs(D_E[a, remaining] - target_dE)
        order2 = np.lexsort((orig_positions[remaining], gaps))
        matched_idx = remaining[order2[:20]]

        retrieved_dF[a] = D_F[a, bG_local]
        matched_dF_mean[a] = np.mean(D_F[a, matched_idx])
        matched_gap[a] = np.mean(np.abs(D_E[a, matched_idx] - target_dE))
        bG_of[a] = bG_local
        evaluable[a] = True

    ev_idx = np.where(evaluable)[0]
    n_eval = len(ev_idx)
    r_dF = retrieved_dF[ev_idx]
    m_dF = matched_dF_mean[ev_idx]
    gap = matched_gap[ev_idx]
    win = (r_dF < m_dF).astype(float)

    def stat_ratio(idx):
        return np.mean(r_dF[idx]) / np.mean(m_dF[idx])

    def stat_win(idx):
        return np.mean(win[idx])

    def stat_gap_ratio(idx):
        return np.mean(gap[idx]) / e090

    point_r, lo_r, hi_r, _ = bootstrap_ci_indices(n_eval, seed=101004, n_boot=3000, compute_stat_fn=stat_ratio)
    point_w, lo_w, hi_w, _ = bootstrap_ci_indices(n_eval, seed=101004, n_boot=3000, compute_stat_fn=stat_win)
    point_g, lo_g, hi_g, _ = bootstrap_ci_indices(n_eval, seed=101005, n_boot=3000, compute_stat_fn=stat_gap_ratio)

    return {
        "e090": e090,
        "n_evaluable_anchors": int(n_eval),
        "retrieval_ratio": point_r,
        "retrieval_ratio_ci95_lower": lo_r,
        "retrieval_ratio_ci95_upper": hi_r,
        "retrieval_win_rate": point_w,
        "retrieval_win_rate_ci95_lower": lo_w,
        "retrieval_win_rate_ci95_upper": hi_w,
        "raw_control_matching_gap_ratio": point_g,
        "raw_control_matching_gap_ratio_ci95_lower": lo_g,
        "raw_control_matching_gap_ratio_ci95_upper": hi_g,
        "gate2_pass": bool(hi_r < 0.70),
        "gate3_pass": bool(lo_w > 0.80),
        "raw_control_matching_pass": bool(point_g <= 0.10 and hi_g < 0.10),
        "evaluable_mask": evaluable,
        "retrieved_dF": retrieved_dF,
        "matched_dF_mean": matched_dF_mean,
        "matched_gap": matched_gap,
        "bG_of": bG_of,
    }
