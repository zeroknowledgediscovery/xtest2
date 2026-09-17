"""TEST 1-4 implementations, exactly as specified in instructions.md."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

from . import config
from .bootstrap import bootstrap_ci


# ---------------------------------------------------------------------------
# Eval cohort
# ---------------------------------------------------------------------------

def select_eval_cohort(feat, train_mask):
    blind_numeric = feat[config.BLIND_COLS]
    eligible_pos = np.flatnonzero((~train_mask) & blind_numeric.notna().all(axis=1).to_numpy())
    eval_rng = np.random.default_rng(config.EVAL_COHORT_SEED)
    eval_pos = eval_rng.choice(eligible_pos, size=config.EVAL_COHORT_SIZE, replace=False)
    return eval_pos


# ---------------------------------------------------------------------------
# TEST 1 - geometry novelty
# ---------------------------------------------------------------------------

def _topk_neighbors(D, k):
    n = D.shape[0]
    Dx = D.copy()
    np.fill_diagonal(Dx, np.inf)
    Dx[np.isnan(Dx)] = np.inf
    idx = np.argsort(Dx, axis=1)[:, :k]
    return idx


def _jaccard_overlap(idx_a, idx_b):
    n, k = idx_a.shape
    j = np.empty(n)
    for i in range(n):
        j[i] = len(set(idx_a[i]) & set(idx_b[i])) / k
    return j


def _anchorwise_spearman(D_g, D_x):
    n = D_g.shape[0]
    rho = np.full(n, np.nan)
    for i in range(n):
        g = D_g[i].copy()
        x = D_x[i].copy()
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask &= np.isfinite(x) & np.isfinite(g)
        if mask.sum() >= 3:
            r, _ = spearmanr(g[mask], x[mask])
            rho[i] = r
    return rho


def run_test1(D_G, D_E, D_M):
    n = D_G.shape[0]
    idxG = _topk_neighbors(D_G, config.N_NEIGHBORS)
    idxE = _topk_neighbors(D_E, config.N_NEIGHBORS)
    idxM = _topk_neighbors(D_M, config.N_NEIGHBORS)

    J_E = _jaccard_overlap(idxG, idxE)
    J_M = _jaccard_overlap(idxG, idxM)

    seed = config.GEOMETRY_NOVELTY_BOOTSTRAP_SEED
    nboot = config.N_ANCHOR_BOOTSTRAP
    mJE, mJE_lo, mJE_hi, _ = bootstrap_ci(n, seed, nboot, (J_E,), lambda a: np.mean(a, axis=-1))
    mJM, mJM_lo, mJM_hi, _ = bootstrap_ci(n, seed, nboot, (J_M,), lambda a: np.mean(a, axis=-1))

    iu = np.triu_indices(n, k=1)
    maskE = np.isfinite(D_E[iu])
    rho_GE, _ = spearmanr(D_G[iu][maskE], D_E[iu][maskE])
    maskM = np.isfinite(D_M[iu])
    rho_GM, _ = spearmanr(D_G[iu][maskM], D_M[iu][maskM])

    rhoE_anchor = _anchorwise_spearman(D_G, D_E)
    rhoM_anchor = _anchorwise_spearman(D_G, D_M)
    okE = np.isfinite(rhoE_anchor)
    okM = np.isfinite(rhoM_anchor)
    absE = np.abs(rhoE_anchor[okE])
    absM = np.abs(rhoM_anchor[okM])
    _, _, ciE_hi, _ = bootstrap_ci(len(absE), seed, nboot, (absE,), lambda a: np.mean(a, axis=-1))
    _, _, ciM_hi, _ = bootstrap_ci(len(absM), seed, nboot, (absM,), lambda a: np.mean(a, axis=-1))

    j20_pass = (mJE_hi < 0.50) and (mJM_hi < 0.50)
    rank_pass = (abs(rho_GE) < 0.75) and (abs(rho_GM) < 0.75) and (ciE_hi < 0.75) and (ciM_hi < 0.75)
    geometry_novel = bool(j20_pass and rank_pass)

    return {
        "geometry_novel": geometry_novel,
        "mean_J20_euclidean": mJE, "mean_J20_euclidean_ci95_lower": mJE_lo, "mean_J20_euclidean_ci95_upper": mJE_hi,
        "mean_J20_mixed": mJM, "mean_J20_mixed_ci95_lower": mJM_lo, "mean_J20_mixed_ci95_upper": mJM_hi,
        "global_rho_geometry_euclidean": float(rho_GE), "global_rho_geometry_mixed": float(rho_GM),
        "anchorwise_abs_rho_euclidean_ci95_upper": float(ciE_hi),
        "anchorwise_abs_rho_mixed_ci95_upper": float(ciM_hi),
    }


# ---------------------------------------------------------------------------
# TEST 2 - blind distant-analog retrieval
# ---------------------------------------------------------------------------

def run_test2(D_G, D_E, blind_std):
    """blind_std: (n,8) standardized transport values for the eval cohort."""
    n = D_G.shape[0]
    d_F = np.sqrt(np.mean((blind_std[:, None, :] - blind_std[None, :, :]) ** 2, axis=-1))

    iu = np.triu_indices(n, k=1)
    e90 = float(np.nanquantile(D_E[iu], 0.90))

    retrieved_dF = np.full(n, np.nan)
    control_mean_dF = np.full(n, np.nan)
    control_mean_gap = np.full(n, np.nan)
    evaluable = np.zeros(n, dtype=bool)

    for a in range(n):
        de_row = D_E[a].copy()
        de_row[a] = np.nan
        far_mask = np.isfinite(de_row) & (de_row >= e90)
        cand = np.flatnonzero(far_mask)
        if len(cand) < config.N_CONTROLS + 1:
            continue
        dg_row = D_G[a][cand]
        b_g = cand[np.argmin(dg_row)]
        de_bg = de_row[b_g]

        others = cand[cand != b_g]
        if len(others) < config.N_CONTROLS:
            continue
        gap = np.abs(de_row[others] - de_bg)
        order = np.argsort(gap)[:config.N_CONTROLS]
        controls = others[order]

        evaluable[a] = True
        retrieved_dF[a] = d_F[a, b_g]
        control_mean_dF[a] = np.mean(d_F[a, controls])
        control_mean_gap[a] = np.mean(gap[order])

    n_eval = int(evaluable.sum())
    ret_dF = retrieved_dF[evaluable]
    ctl_dF = control_mean_dF[evaluable]
    ctl_gap = control_mean_gap[evaluable]

    seed_r = config.RETRIEVAL_BOOTSTRAP_SEED
    seed_g = config.MATCHING_GAP_BOOTSTRAP_SEED
    nboot = config.N_ANCHOR_BOOTSTRAP

    def ratio_stat(num, den):
        return np.mean(num, axis=-1) / np.mean(den, axis=-1)

    R, R_lo, R_hi, _ = bootstrap_ci(n_eval, seed_r, nboot, (ret_dF, ctl_dF), ratio_stat)

    win = (ret_dF < ctl_dF).astype(float)
    W, W_lo, W_hi, _ = bootstrap_ci(n_eval, seed_r, nboot, (win,), lambda a: np.mean(a, axis=-1))

    def gmatch_stat(gap):
        return np.mean(gap, axis=-1) / e90

    G_match, G_lo, G_hi, _ = bootstrap_ci(n_eval, seed_g, nboot, (ctl_gap,), gmatch_stat)

    matching_pass = bool((G_match <= 0.10) and (G_hi < 0.10))
    retrieval_pass = bool((R <= 0.95) and (R_hi < 1.0) and (W >= 0.60) and (W_lo > 0.55) and matching_pass)

    return {
        "retrieval_useful": retrieval_pass,
        "raw_control_matching_pass": matching_pass,
        "n_evaluable_anchors": n_eval,
        "retrieval_ratio": R, "retrieval_ratio_ci95_lower": R_lo, "retrieval_ratio_ci95_upper": R_hi,
        "retrieval_win_rate": W, "retrieval_win_rate_ci95_lower": W_lo, "retrieval_win_rate_ci95_upper": W_hi,
        "mean_abs_control_raw_distance_gap": float(np.mean(ctl_gap)) if n_eval else float("nan"),
        "raw_control_matching_gap_ratio": G_match,
        "raw_control_matching_gap_ratio_ci95_lower": G_lo,
        "raw_control_matching_gap_ratio_ci95_upper": G_hi,
        "raw_far_threshold_q90": e90,
        "_evaluable_mask": evaluable,
        "_retrieved_dF": retrieved_dF,
        "_control_mean_dF": control_mean_dF,
    }


# ---------------------------------------------------------------------------
# TEST 3 - unconditional generation
# ---------------------------------------------------------------------------

def generate_unconditional_samples(model, n, seed):
    rng = np.random.default_rng(seed)
    Z = model.sample_unconditional_latent(n, rng)
    raw = model.from_latent(Z)
    return pd.DataFrame(raw)


def _row_key_matrix(df, numeric_cols, categorical_cols, decimals=8):
    num = np.round(df[numeric_cols].to_numpy(dtype=float), decimals)
    cat = df[categorical_cols].to_numpy(dtype=object)
    keys = []
    for i in range(len(df)):
        keys.append((tuple(num[i]), tuple(cat[i])))
    return keys


def test3_blank_generation_gate(gen_df, train_df, numeric_cols, categorical_cols):
    panel_cols = numeric_cols + categorical_cols
    n = len(gen_df)
    valid = np.ones(n, dtype=bool)
    for c in panel_cols:
        if c in numeric_cols:
            valid &= np.isfinite(gen_df[c].to_numpy(dtype=float))
        else:
            valid &= gen_df[c].notna().to_numpy()
    valid_count = int(valid.sum())

    gen_keys = _row_key_matrix(gen_df, numeric_cols, categorical_cols)
    unique_count = len(set(gen_keys))

    train_keys = set(_row_key_matrix(train_df, numeric_cols, categorical_cols))
    exact_match_count = sum(1 for k in gen_keys if k in train_keys)

    valid_frac = valid_count / n
    unique_frac = unique_count / n
    exact_frac = exact_match_count / n
    passes = (valid_count >= 9500) and (unique_frac >= 0.80) and (exact_frac <= 0.10)
    return {
        "blank_generation_valid": bool(passes),
        "unconditional_requested_count": n,
        "unconditional_valid_count": valid_count,
        "unconditional_unique_fraction": unique_frac,
        "unconditional_exact_training_match_fraction": exact_frac,
    }


def test3a_marginal_fidelity(gen_df, heldout_df, train_df, numeric_cols, categorical_cols):
    rows = []
    for c in numeric_cols:
        g = gen_df[c].to_numpy(dtype=float)
        h = heldout_df[c].to_numpy(dtype=float)
        h = h[np.isfinite(h)]
        g = g[np.isfinite(g)]
        tr = train_df[c].to_numpy(dtype=float)
        tr = tr[np.isfinite(tr)]
        q75, q25 = np.percentile(tr, [75, 25])
        iqr = q75 - q25
        if iqr <= 0:
            iqr = tr.max() - tr.min()
        if iqr <= 0 or len(h) == 0 or len(g) == 0:
            d = np.nan
        else:
            d = wasserstein_distance(g, h) / iqr
        rows.append({"coordinate": c, "type": "numeric", "fidelity_metric": "W1_over_train_IQR", "value": float(d)})
    for c in categorical_cols:
        g = gen_df[c].dropna()
        h = heldout_df[c].dropna()
        cats = sorted(set(g.unique().tolist()) | set(h.unique().tolist()))
        pg = g.value_counts(normalize=True).reindex(cats, fill_value=0.0)
        ph = h.value_counts(normalize=True).reindex(cats, fill_value=0.0)
        tvd = 0.5 * np.abs(pg.to_numpy() - ph.to_numpy()).sum()
        rows.append({"coordinate": c, "type": "categorical", "fidelity_metric": "total_variation_distance", "value": float(tvd)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TEST 3B - cross-structure retention
# ---------------------------------------------------------------------------

def _encode_features(df, feature_cols, numeric_cols, model):
    n = len(df)
    X = np.full((n, len(feature_cols)), np.nan)
    for j, c in enumerate(feature_cols):
        if c in numeric_cols:
            X[:, j] = df[c].to_numpy(dtype=float)
        else:
            idx_map = model.marginals[c].index
            vals = df[c].tolist()
            X[:, j] = [idx_map.get(v, np.nan) if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan for v in vals]
    return X


def run_test3b_cross_structure(model, numeric_cols, categorical_cols,
                                train_real_10k, gen_10k, ind_10k, heldout_full):
    panel_cols = numeric_cols + categorical_cols
    train_iqr = {}
    for c in numeric_cols:
        v = train_real_10k[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        q75, q25 = np.percentile(v, [75, 25])
        iqr = q75 - q25
        if iqr <= 0:
            iqr = v.max() - v.min()
        train_iqr[c] = iqr if iqr > 0 else 1.0

    per_target_rows = []
    per_row_losses = {}  # target -> {"real":arr,"gen":arr,"ind":arr} aligned to heldout eval rows used

    for target in panel_cols:
        feature_cols = [c for c in panel_cols if c != target]
        is_numeric = target in numeric_cols

        y_train_full = train_real_10k[target]
        y_gen_full = gen_10k[target]
        y_ind_full = ind_10k[target]
        y_heldout_full = heldout_full[target]

        if is_numeric:
            y_train_obs = np.isfinite(y_train_full.to_numpy(dtype=float))
        else:
            y_train_obs = y_train_full.notna().to_numpy()
        X_train = _encode_features(train_real_10k, feature_cols, numeric_cols, model)[y_train_obs]
        y_train = y_train_full.to_numpy()[y_train_obs]

        X_gen = _encode_features(gen_10k, feature_cols, numeric_cols, model)
        y_gen = y_gen_full.to_numpy()
        X_ind = _encode_features(ind_10k, feature_cols, numeric_cols, model)
        y_ind = y_ind_full.to_numpy()

        if is_numeric:
            eval_obs = np.isfinite(y_heldout_full.to_numpy(dtype=float))
        else:
            eval_obs = y_heldout_full.notna().to_numpy()
        X_eval = _encode_features(heldout_full, feature_cols, numeric_cols, model)[eval_obs]
        y_eval = y_heldout_full.to_numpy()[eval_obs]

        if is_numeric:
            if y_train.dtype != float:
                y_train = y_train.astype(float)
            m_real = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3, max_features="sqrt", random_state=config.CROSS_STRUCTURE_VERIFIER_SEED)
            m_gen = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3, max_features="sqrt", random_state=config.CROSS_STRUCTURE_VERIFIER_SEED)
            m_ind = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3, max_features="sqrt", random_state=config.CROSS_STRUCTURE_VERIFIER_SEED)
            m_real.fit(X_train, y_train); m_gen.fit(X_gen, y_gen.astype(float)); m_ind.fit(X_ind, y_ind.astype(float))
            p_real = m_real.predict(X_eval); p_gen = m_gen.predict(X_eval); p_ind = m_ind.predict(X_eval)
            y_eval_f = y_eval.astype(float)
            iqr = train_iqr[target]
            l_real = np.abs(p_real - y_eval_f) / iqr
            l_gen = np.abs(p_gen - y_eval_f) / iqr
            l_ind = np.abs(p_ind - y_eval_f) / iqr
        else:
            idx_map = model.marginals[target].index
            cats = model.marginals[target].categories

            def enc_y(y):
                return np.array([idx_map.get(v, -1) for v in y])
            yt = enc_y(y_train); yg = enc_y(y_gen); yi = enc_y(y_ind); ye = enc_y(y_eval)
            m_real = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=3, max_features="sqrt", random_state=config.CROSS_STRUCTURE_VERIFIER_SEED)
            m_gen = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=3, max_features="sqrt", random_state=config.CROSS_STRUCTURE_VERIFIER_SEED)
            m_ind = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=3, max_features="sqrt", random_state=config.CROSS_STRUCTURE_VERIFIER_SEED)
            m_real.fit(X_train, yt); m_gen.fit(X_gen, yg); m_ind.fit(X_ind, yi)
            p_real = m_real.predict(X_eval); p_gen = m_gen.predict(X_eval); p_ind = m_ind.predict(X_eval)
            l_real = (p_real != ye).astype(float)
            l_gen = (p_gen != ye).astype(float)
            l_ind = (p_ind != ye).astype(float)

        L_real, L_gen, L_ind = float(np.mean(l_real)), float(np.mean(l_gen)), float(np.mean(l_ind))
        informative = L_real <= 0.95 * L_ind
        per_target_rows.append({
            "target": target, "type": "numeric" if is_numeric else "categorical",
            "n_eval_rows": len(y_eval), "L_real": L_real, "L_gen": L_gen, "L_ind": L_ind,
            "informative": bool(informative),
        })
        per_row_losses[target] = {"real": l_real, "gen": l_gen, "ind": l_ind, "informative": bool(informative)}

    by_target_df = pd.DataFrame(per_target_rows)

    informative_targets = [t for t in panel_cols if per_row_losses[t]["informative"]]
    g_vals = []
    win = 0
    for t in informative_targets:
        d = per_row_losses[t]
        denom = np.mean(d["ind"]) - np.mean(d["real"])
        g = (np.mean(d["ind"]) - np.mean(d["gen"])) / denom if denom != 0 else np.nan
        g_vals.append(g)
        if np.mean(d["gen"]) < np.mean(d["ind"]):
            win += 1
    G_point = float(np.nanmean(g_vals)) if g_vals else float("nan")
    win_frac = win / len(informative_targets) if informative_targets else float("nan")

    # bootstrap over held-out ROWS, without refitting: resample rows per-target
    # (each target has its own eval-row subset; we bootstrap by resampling a
    # common set of bootstrap draws over row positions 0..len-1 per target).
    rng = np.random.default_rng(config.CROSS_STRUCTURE_BOOTSTRAP_SEED)
    nboot = 2000
    G_boot = np.empty(nboot)
    for b in range(nboot):
        gs = []
        for t in informative_targets:
            d = per_row_losses[t]
            n_t = len(d["real"])
            idx = rng.integers(0, n_t, size=n_t)
            l_real_b = d["real"][idx].mean()
            l_gen_b = d["gen"][idx].mean()
            l_ind_b = d["ind"][idx].mean()
            denom = l_ind_b - l_real_b
            gs.append((l_ind_b - l_gen_b) / denom if denom != 0 else np.nan)
        G_boot[b] = np.nanmean(gs) if gs else np.nan

    G_lo = float(np.nanpercentile(G_boot, 2.5))
    G_hi = float(np.nanpercentile(G_boot, 97.5))

    cross_structure_preserved = bool((G_lo > 0) and (win_frac >= 0.60))

    return {
        "cross_structure_preserved": cross_structure_preserved,
        "n_cross_structure_targets": len(informative_targets),
        "cross_structure_retention_G": G_point,
        "cross_structure_retention_G_ci95_lower": G_lo,
        "cross_structure_retention_G_ci95_upper": G_hi,
        "cross_structure_target_win_fraction": win_frac,
        "by_target": by_target_df,
    }


# ---------------------------------------------------------------------------
# TEST 4 - arbitrary partial-state completion
# ---------------------------------------------------------------------------

def _median_mode_completion(draws_raw_by_col, numeric_cols, categorical_cols):
    """draws_raw_by_col: dict col -> array (n_draws, n_rows) raw values.
    Returns dict col -> array (n_rows,) point completion."""
    out = {}
    for c in numeric_cols:
        out[c] = np.median(draws_raw_by_col[c].astype(float), axis=0)
    for c in categorical_cols:
        arr = draws_raw_by_col[c]
        n_draws, n_rows = arr.shape
        modes = np.empty(n_rows, dtype=object)
        for i in range(n_rows):
            vals, counts = np.unique(arr[:, i], return_counts=True)
            best = counts.max()
            cand = sorted(vals[counts == best])
            modes[i] = cand[0]
        out[c] = modes
    return out


def build_completion_masks(feat, panel_numeric, panel_categorical, model, eligible_pos):
    panel_cols = panel_numeric + panel_categorical
    row_rng = np.random.default_rng(config.COMPLETION_ROW_SEED)
    sel_pos = row_rng.choice(eligible_pos, size=config.COMPLETION_ROW_COUNT, replace=False)
    rows = feat.iloc[sel_pos][panel_cols].reset_index(drop=True)

    Z_true = model.to_latent(rows)
    obs_mask_true = np.isfinite(Z_true)

    mask_rng = np.random.default_rng(config.COMPLETION_MASK_SEED)
    n, p = Z_true.shape
    hidden_masks = {}
    for q in (0.20, 0.50, 0.80):
        hidden = np.zeros((n, p), dtype=bool)
        for i in range(n):
            obs_idx = np.flatnonzero(obs_mask_true[i])
            n_hide = int(round(q * len(obs_idx)))
            if n_hide > 0:
                chosen = mask_rng.choice(obs_idx, size=n_hide, replace=False)
                hidden[i, chosen] = True
        hidden_masks[q] = hidden
    return sel_pos, rows, Z_true, obs_mask_true, hidden_masks, mask_rng


def run_test4(model, baseline, feat, panel_numeric, panel_categorical, train_df):
    panel_cols = panel_numeric + panel_categorical
    n_obs_frac = feat[panel_cols].notna().mean(axis=1)
    eligible_pos = np.flatnonzero(n_obs_frac.to_numpy() >= 0.80)

    sel_pos, rows, Z_true, obs_mask_true, hidden_masks, mask_rng = build_completion_masks(
        feat, panel_numeric, panel_categorical, model, eligible_pos)
    n = len(rows)

    train_iqr = {}
    for c in panel_numeric:
        v = train_df[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        q75, q25 = np.percentile(v, [75, 25])
        iqr = q75 - q25
        train_iqr[c] = iqr if iqr > 0 else (v.max() - v.min() if v.max() > v.min() else 1.0)

    results = {}
    coord_rows = []

    for q, hidden in hidden_masks.items():
        Z_template = np.where(hidden, np.nan, Z_true)
        draws_latent = model.sample_conditional_latent(Z_template, mask_rng, n_draws=10)  # (10,n,p)
        raw_draws = {c: np.empty((10, n), dtype=object if c in panel_categorical else float) for c in panel_cols}
        for d in range(10):
            raw_d = model.from_latent(draws_latent[d])
            for c in panel_cols:
                raw_draws[c][d] = raw_d[c]
        model_point = _median_mode_completion(raw_draws, panel_numeric, panel_categorical)

        observed_dict = {}
        hidden_mask_per_col = {}
        for j, c in enumerate(panel_cols):
            vals = rows[c].to_numpy(dtype=object).copy()
            vals[hidden[:, j]] = None
            observed_dict[c] = vals
            hidden_mask_per_col[c] = hidden[:, j]

        ind_draws = {c: np.empty((10, n), dtype=object if c in panel_categorical else float) for c in panel_cols}
        for d in range(10):
            samp = baseline.sample_conditional(observed_dict, hidden_mask_per_col, n, mask_rng)
            for c in panel_cols:
                ind_draws[c][d] = samp[c]
        ind_point = _median_mode_completion(ind_draws, panel_numeric, panel_categorical)

        loss_model = np.full((n, len(panel_cols)), np.nan)
        loss_ind = np.full((n, len(panel_cols)), np.nan)
        for j, c in enumerate(panel_cols):
            hj = hidden[:, j]
            if not hj.any():
                continue
            true_vals = rows[c].to_numpy(dtype=object)
            if c in panel_numeric:
                tv = true_vals[hj].astype(float)
                lm = np.abs(model_point[c][hj] - tv) / train_iqr[c]
                li = np.abs(ind_point[c][hj] - tv) / train_iqr[c]
            else:
                tv = true_vals[hj]
                lm = (model_point[c][hj] != tv).astype(float)
                li = (ind_point[c][hj] != tv).astype(float)
            loss_model[hj, j] = lm
            loss_ind[hj, j] = li

        rng_boot = np.random.default_rng(config.COMPLETION_BOOTSTRAP_SEED)
        nboot = config.N_ANCHOR_BOOTSTRAP
        idx = rng_boot.integers(0, n, size=(nboot, n))

        def level_L(loss_mat, idxmat):
            # Column-at-a-time to avoid materializing an (nboot, n, p) array.
            nboot_, p_ = idxmat.shape[0], loss_mat.shape[1]
            colmeans = np.full((nboot_, p_), np.nan)
            for j in range(p_):
                col = loss_mat[:, j]
                mask_j = np.isfinite(col)
                if not mask_j.any():
                    continue
                filled_j = np.where(mask_j, col, 0.0)
                sum_r = filled_j[idxmat].sum(axis=1)
                count_r = mask_j[idxmat].sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    colmeans[:, j] = np.where(count_r > 0, sum_r / count_r, np.nan)
            with np.errstate(invalid="ignore"):
                return np.nanmean(colmeans, axis=1)  # (nboot,)

        L_M_boot = level_L(loss_model, idx)
        L_I_boot = level_L(loss_ind, idx)
        with np.errstate(invalid="ignore", divide="ignore"):
            R_boot = L_M_boot / L_I_boot

        with np.errstate(invalid="ignore"):
            colmean_m = np.nanmean(loss_model, axis=0)
            colmean_i = np.nanmean(loss_ind, axis=0)
        L_M_point = float(np.nanmean(colmean_m))
        L_I_point = float(np.nanmean(colmean_i))
        R_point = L_M_point / L_I_point if L_I_point != 0 else np.nan
        R_lo = float(np.nanpercentile(R_boot, 2.5))
        R_hi = float(np.nanpercentile(R_boot, 97.5))

        evaluable_cols = ~np.isnan(colmean_m) & ~np.isnan(colmean_i)
        win_frac = float(np.mean(colmean_m[evaluable_cols] < colmean_i[evaluable_cols])) if evaluable_cols.any() else float("nan")

        for j, c in enumerate(panel_cols):
            if evaluable_cols[j]:
                coord_rows.append({
                    "masking_level": q, "coordinate": c,
                    "type": "numeric" if c in panel_numeric else "categorical",
                    "L_model": float(colmean_m[j]), "L_independent": float(colmean_i[j]),
                    "model_wins": bool(colmean_m[j] < colmean_i[j]),
                })

        results[q] = {
            "R": R_point, "R_lo": R_lo, "R_hi": R_hi, "win_frac": win_frac,
            "L_M": L_M_point, "L_I": L_I_point,
        }

    passes = (
        results[0.50]["R_hi"] < 1.0 and results[0.80]["R_hi"] < 1.0 and
        results[0.50]["win_frac"] >= 0.60 and results[0.80]["win_frac"] >= 0.60
    )

    coord_df = pd.DataFrame(coord_rows)
    out = {
        "conditional_completion_useful": bool(passes),
        "completion_ratio_hidden20": results[0.20]["R"],
        "completion_ratio_hidden20_ci95_lower": results[0.20]["R_lo"],
        "completion_ratio_hidden20_ci95_upper": results[0.20]["R_hi"],
        "completion_ratio_hidden50": results[0.50]["R"],
        "completion_ratio_hidden50_ci95_lower": results[0.50]["R_lo"],
        "completion_ratio_hidden50_ci95_upper": results[0.50]["R_hi"],
        "completion_ratio_hidden80": results[0.80]["R"],
        "completion_ratio_hidden80_ci95_lower": results[0.80]["R_lo"],
        "completion_ratio_hidden80_ci95_upper": results[0.80]["R_hi"],
        "completion_coordinate_win_fraction_hidden50": results[0.50]["win_frac"],
        "completion_coordinate_win_fraction_hidden80": results[0.80]["win_frac"],
        "by_coordinate": coord_df,
    }
    return out
