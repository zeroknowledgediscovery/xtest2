"""TEST 4 -- partial-state completion (Strong Gate 5)."""
import numpy as np
import pandas as pd

MASK_LEVELS = [0.05, 0.10, 0.20, 0.50]
MASK_SELECT_SEED = 27182818
MODEL_SAMPLE_SEED = 90909090
IND_SAMPLE_SEED = 314159
N_REPS = 10
COHORT_SIZE = 8000
COHORT_SEED = 20260916


def select_cohort(held_panel_df, panel_cols):
    n_panel = len(panel_cols)
    obs_count = held_panel_df[panel_cols].notna().sum(axis=1)
    frac = obs_count / n_panel
    eligible = held_panel_df.index[frac >= 0.80].to_numpy()
    rng = np.random.default_rng(COHORT_SEED)
    chosen = rng.choice(eligible, size=COHORT_SIZE, replace=False)
    return chosen


def build_obs_lists(held_panel_df, chosen_idx, panel_cols):
    obs_lists = []
    for idx in chosen_idx:
        row = held_panel_df.loc[idx]
        obs = [c for c in panel_cols if pd.notna(row[c])]
        obs_lists.append(obs)
    return obs_lists


def build_hidden_sets(obs_lists, panel_cols):
    """Single shared RNG consumed sequentially over (row, level) pairs in
    that nested order, as specified."""
    rng = np.random.default_rng(MASK_SELECT_SEED)
    n_rows = len(obs_lists)
    hidden = {lvl: [None] * n_rows for lvl in MASK_LEVELS}
    for i in range(n_rows):
        obs = obs_lists[i]
        for lvl in MASK_LEVELS:
            k = int(round(lvl * len(obs)))
            perm = rng.permutation(len(obs))
            hidden_names = [obs[p] for p in perm[:k]]
            hidden[lvl][i] = hidden_names
    return hidden


def aggregate_reps(values, is_numeric):
    if is_numeric:
        return np.median(values, axis=1)
    else:
        out = np.empty(values.shape[0], dtype=object)
        for i in range(values.shape[0]):
            vals, counts = np.unique(values[i], return_counts=True)
            maxc = counts.max()
            candidates = sorted(vals[counts == maxc].tolist())
            out[i] = candidates[0]
        return out


def compute_losses(true_vals, pred_vals, is_numeric, iqr):
    if is_numeric:
        return np.abs(pred_vals.astype(float) - true_vals.astype(float)) / iqr
    else:
        return (pred_vals.astype(str) != true_vals.astype(str)).astype(float)


def run_test4(model, ind_model, held_panel_df, numeric_cols, categorical_cols,
              train_iqr, n_boot=3000, boot_seed=577215):
    panel_cols = numeric_cols + categorical_cols
    chosen_idx = select_cohort(held_panel_df, panel_cols)
    cohort_df = held_panel_df.loc[chosen_idx].reset_index(drop=True)
    obs_lists = build_obs_lists(held_panel_df, chosen_idx, panel_cols)
    hidden = build_hidden_sets(obs_lists, panel_cols)

    n_rows = len(cohort_df)
    n_coords = len(panel_cols)
    coord_index = {c: i for i, c in enumerate(panel_cols)}

    loss_M = {lvl: np.full((n_rows, n_coords), np.nan) for lvl in MASK_LEVELS}
    loss_IND = {lvl: np.full((n_rows, n_coords), np.nan) for lvl in MASK_LEVELS}

    model_rng_seed = MODEL_SAMPLE_SEED
    ind_rng_seed = IND_SAMPLE_SEED

    rows_records = []

    for lvl in MASK_LEVELS:
        hidden_mask_df = pd.DataFrame(False, index=range(n_rows), columns=panel_cols)
        for i in range(n_rows):
            for c in hidden[lvl][i]:
                hidden_mask_df.at[i, c] = True

        out_num_M, out_cat_M = model.sample_conditional_batch(
            cohort_df, hidden_mask_df, n_reps=N_REPS, seed=model_rng_seed + int(lvl * 1000))
        out_num_I, out_cat_I = ind_model_sample_conditional(
            ind_model, cohort_df, hidden_mask_df, n_reps=N_REPS,
            seed=ind_rng_seed + int(lvl * 1000), numeric_cols=numeric_cols,
            categorical_cols=categorical_cols)

        for c in numeric_cols:
            mask_col = hidden_mask_df[c].to_numpy()
            if not mask_col.any():
                continue
            pred_M = aggregate_reps(out_num_M[c][mask_col], True)
            pred_I = aggregate_reps(out_num_I[c][mask_col], True)
            true_vals = cohort_df.loc[mask_col, c].to_numpy(dtype=float)
            iqr = train_iqr[c]
            lM = compute_losses(true_vals, pred_M, True, iqr)
            lI = compute_losses(true_vals, pred_I, True, iqr)
            rows_idx = np.where(mask_col)[0]
            loss_M[lvl][rows_idx, coord_index[c]] = lM
            loss_IND[lvl][rows_idx, coord_index[c]] = lI
            for ridx, lmv, liv in zip(rows_idx, lM, lI):
                rows_records.append({"masking_level": lvl, "row": int(ridx), "coordinate": c,
                                      "loss_model": lmv, "loss_ind": liv})

        for c in categorical_cols:
            mask_col = hidden_mask_df[c].to_numpy()
            if not mask_col.any():
                continue
            pred_M = aggregate_reps(out_cat_M[c][mask_col], False)
            pred_I = aggregate_reps(out_cat_I[c][mask_col], False)
            true_vals = cohort_df.loc[mask_col, c].to_numpy()
            lM = compute_losses(true_vals, pred_M, False, None)
            lI = compute_losses(true_vals, pred_I, False, None)
            rows_idx = np.where(mask_col)[0]
            loss_M[lvl][rows_idx, coord_index[c]] = lM
            loss_IND[lvl][rows_idx, coord_index[c]] = lI
            for ridx, lmv, liv in zip(rows_idx, lM, lI):
                rows_records.append({"masking_level": lvl, "row": int(ridx), "coordinate": c,
                                      "loss_model": lmv, "loss_ind": liv})

    def level_stat(L, idx):
        sub = L[idx, :]
        with np.errstate(invalid="ignore"):
            colmeans = np.nanmean(sub, axis=0)
        return np.nanmean(colmeans)

    results = {}
    boot_rng = np.random.default_rng(boot_seed)
    idx_all = np.arange(n_rows)
    for lvl in MASK_LEVELS:
        LM = loss_M[lvl]
        LI = loss_IND[lvl]
        point_M = level_stat(LM, idx_all)
        point_I = level_stat(LI, idx_all)
        point_R = point_M / point_I

        boots = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = boot_rng.choice(idx_all, size=n_rows, replace=True)
            mM = level_stat(LM, idx)
            mI = level_stat(LI, idx)
            boots[b] = mM / mI
        lo = np.nanpercentile(boots, 2.5)
        hi = np.nanpercentile(boots, 97.5)

        coord_win = {}
        for c in panel_cols:
            j = coord_index[c]
            colM = LM[:, j]
            colI = LI[:, j]
            valid = ~np.isnan(colM) & ~np.isnan(colI)
            if valid.sum() == 0:
                continue
            coord_win[c] = float(np.mean(colM[valid] < colI[valid]))

        results[lvl] = {
            "R_C": point_R, "ci_lower": lo, "ci_upper": hi,
            "L_M": point_M, "L_IND": point_I,
            "coord_win_fraction": np.mean(list(coord_win.values())) if coord_win else np.nan,
            "coord_win_detail": coord_win,
        }

    by_coord_df = pd.DataFrame(rows_records)
    return results, by_coord_df


def ind_model_sample_conditional(ind_model, obs_df, hidden_mask_df, n_reps, seed,
                                  numeric_cols, categorical_cols):
    rng = np.random.default_rng(seed)
    m = len(obs_df)
    out_num = {c: np.empty((m, n_reps)) for c in numeric_cols}
    out_cat = {c: np.empty((m, n_reps), dtype=object) for c in categorical_cols}
    for c in numeric_cols:
        mask_col = hidden_mask_df[c].to_numpy()
        if not mask_col.any():
            continue
        for r in range(n_reps):
            out_num[c][:, r] = ind_model.sample_column(c, m, rng)
    for c in categorical_cols:
        mask_col = hidden_mask_df[c].to_numpy()
        if not mask_col.any():
            continue
        for r in range(n_reps):
            out_cat[c][:, r] = ind_model.sample_column(c, m, rng)
    return out_num, out_cat
