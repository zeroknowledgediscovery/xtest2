"""TEST 4 - arbitrary partial-state completion, for the submitted model and
the independent-marginal baseline, using identical masks.
"""
import numpy as np
import pandas as pd

import config
from generation import decode_samples

N_REPLICATES = 10
LEVELS = [0.20, 0.50, 0.80]


def select_completion_rows(coord_frame: pd.DataFrame, test_positions: np.ndarray, panel_cols):
    sub = coord_frame.loc[test_positions]
    n_panel = len(panel_cols)

    obs_frac = sub[panel_cols].apply(
        lambda row: sum(
            (row[c] != config.MISSING_STATE) if isinstance(row[c], str) else pd.notna(row[c])
            for c in panel_cols
        ), axis=1
    ) / n_panel
    eligible_local = np.flatnonzero(obs_frac.to_numpy() >= 0.80)
    rng = np.random.default_rng(config.SEED_COMPLETION_ROW_SAMPLE)
    chosen_local = rng.choice(eligible_local, size=min(2000, len(eligible_local)), replace=False)
    return test_positions[chosen_local]


def build_masks(coord_frame: pd.DataFrame, rows: np.ndarray, panel_cols):
    """Returns dict level -> list over rows of (observed_panel_cols_set, hidden_panel_cols_list)."""
    rng = np.random.default_rng(config.SEED_COMPLETION_MASKS)
    masks = {lvl: [] for lvl in LEVELS}
    for r in rows:
        row = coord_frame.loc[r]
        present = [c for c in panel_cols if (row[c] != config.MISSING_STATE if isinstance(row[c], str) else pd.notna(row[c]))]
        for lvl in LEVELS:
            n_hide = int(round(lvl * len(present)))
            n_hide = min(n_hide, len(present))
            hide_idx = rng.choice(len(present), size=n_hide, replace=False) if n_hide > 0 else np.array([], dtype=int)
            hidden = [present[i] for i in hide_idx]
            observed = [c for c in present if c not in set(hidden)]
            masks[lvl].append((observed, hidden))
    return masks


def run_completion_for_model_or_ind(model_sampler, coord_frame, discrete_full, columns, quantizers,
                                     rows, masks_for_level, numeric_panel, categorical_panel, seed,
                                     iqr_train):
    """model_sampler(evidence, n_rep, rng) -> (n_rep, n_cols) discrete array, or
    None to signal 'use independent-marginal' behaviour (handled by caller)."""
    numeric_panel_set = set(numeric_panel)
    col_index = {c: i for i, c in enumerate(columns)}
    loss = {c: np.full(len(rows), np.nan) for c in numeric_panel + categorical_panel}
    rng = np.random.default_rng(seed)

    for ridx, r in enumerate(rows):
        observed, hidden = masks_for_level[ridx]
        if not hidden:
            continue
        evidence = {}
        for c in columns:
            if c in numeric_panel_set or c in categorical_panel:
                if c in observed:
                    evidence[col_index[c]] = int(discrete_full.at[r, c])
                elif c in hidden:
                    continue
                else:
                    evidence[col_index[c]] = int(discrete_full.at[r, c])
            else:
                evidence[col_index[c]] = int(discrete_full.at[r, c])
        draws = model_sampler(evidence, N_REPLICATES, rng)
        hidden_numeric = [c for c in hidden if c in numeric_panel_set]
        hidden_categorical = [c for c in hidden if c not in numeric_panel_set]
        decoded = decode_samples(draws, columns, quantizers, rng, hidden_numeric, hidden_categorical)

        true_row = coord_frame.loc[r]
        for c in hidden:
            if c in numeric_panel_set:
                pred = float(np.median(decoded[c].to_numpy(dtype=float)))
                true_v = float(true_row[c])
                loss[c][ridx] = abs(pred - true_v) / iqr_train[c]
            else:
                vals = decoded[c].astype(str).tolist()
                counts = pd.Series(vals).value_counts()
                maxcount = counts.max()
                modal = sorted([v for v, cnt in counts.items() if cnt == maxcount])[0]
                loss[c][ridx] = float(modal != str(true_row[c]))
    return loss


def run_completion_ind(coord_frame, train_frame, rows, masks_for_level,
                        numeric_panel, categorical_panel, seed, iqr_train):
    numeric_panel_set = set(numeric_panel)
    loss = {c: np.full(len(rows), np.nan) for c in numeric_panel + categorical_panel}
    rng = np.random.default_rng(seed)

    pools_num = {c: train_frame[c].dropna().to_numpy(dtype=float) for c in numeric_panel}
    pools_cat = {c: train_frame[c][train_frame[c] != config.MISSING_STATE].to_numpy() for c in categorical_panel}

    for ridx, r in enumerate(rows):
        _observed, hidden = masks_for_level[ridx]
        if not hidden:
            continue
        true_row = coord_frame.loc[r]
        for c in hidden:
            if c in numeric_panel_set:
                pool = pools_num[c]
                draws = pool[rng.integers(0, len(pool), size=N_REPLICATES)]
                pred = float(np.median(draws))
                true_v = float(true_row[c])
                loss[c][ridx] = abs(pred - true_v) / iqr_train[c]
            else:
                pool = pools_cat[c]
                draws = pool[rng.integers(0, len(pool), size=N_REPLICATES)]
                counts = pd.Series(draws.astype(str)).value_counts()
                maxcount = counts.max()
                modal = sorted([v for v, cnt in counts.items() if cnt == maxcount])[0]
                loss[c][ridx] = float(modal != str(true_row[c]))
    return loss


def summarize_completion(loss_model: dict, loss_ind: dict, panel_cols, n_boot, seed):
    n_rows = len(next(iter(loss_model.values())))
    model_mat = np.column_stack([loss_model[c] for c in panel_cols])
    ind_mat = np.column_stack([loss_ind[c] for c in panel_cols])

    def coord_means(mat, idx):
        sub = mat[idx]
        with np.errstate(invalid="ignore"):
            means = np.nanmean(sub, axis=0)
        return means

    point_model = coord_means(model_mat, np.arange(n_rows))
    point_ind = coord_means(ind_mat, np.arange(n_rows))
    evaluable = ~np.isnan(point_model) & ~np.isnan(point_ind)
    L_M = float(np.nanmean(point_model[evaluable]))
    L_IND = float(np.nanmean(point_ind[evaluable]))
    R = L_M / L_IND if L_IND != 0 else np.nan
    win_fraction = float(np.mean(point_model[evaluable] < point_ind[evaluable])) if evaluable.any() else 0.0

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        m = coord_means(model_mat, idx)
        i = coord_means(ind_mat, idx)
        ev = ~np.isnan(m) & ~np.isnan(i)
        boots[b] = np.nanmean(m[ev]) / np.nanmean(i[ev]) if ev.any() else np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])

    return {
        "R_C": R, "R_C_ci95_lower": float(lo), "R_C_ci95_upper": float(hi),
        "L_M": L_M, "L_IND": L_IND,
        "coordinate_win_fraction": win_fraction,
        "n_evaluable_coordinates": int(evaluable.sum()),
        "per_coordinate": pd.DataFrame({
            "coordinate": panel_cols, "L_M": point_model, "L_IND": point_ind,
        }),
    }
