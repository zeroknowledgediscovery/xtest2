"""TEST 3B -- cross-structure retention (Strong Gate 4).

Uses the exact verifier hyperparameters mandated by the task spec. These
ExtraTrees verifiers are the grading harness's evaluation instrument, not a
second structural model: they never touch d_G or sample(); they only score
the frozen model's already-generated outputs against REAL and IND
comparators, as the spec itself directs.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

REAL_VERIFIER_SUBSAMPLE_SEED = 271828
VERIFIER_RANDOM_STATE = 99173


def fit_predictor_preprocessing(train_panel_df, numeric_cols, categorical_cols):
    stats = {"num_median": {}, "num_iqr": {}, "cat_categories": {}, "cat_map": {}}
    for c in numeric_cols:
        v = train_panel_df[c].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        stats["num_median"][c] = float(np.median(v))
        q75, q25 = np.percentile(v, [75, 25])
        iqr = q75 - q25
        if iqr <= 0:
            rng_ = v.max() - v.min()
            iqr = rng_ if rng_ > 0 else 1.0
        stats["num_iqr"][c] = float(iqr)
    for c in categorical_cols:
        cats = sorted(train_panel_df[c].dropna().unique().tolist())
        stats["cat_categories"][c] = cats
        stats["cat_map"][c] = {v: i for i, v in enumerate(cats)}
    return stats


def build_design(df, predictor_cols, numeric_cols, categorical_cols, stats):
    n = len(df)
    cols = []
    for c in predictor_cols:
        if c in numeric_cols:
            v = df[c].to_numpy(dtype=float)
            med = stats["num_median"][c]
            v = np.where(np.isnan(v), med, v)
            cols.append(v.reshape(-1, 1))
        else:
            mapping = stats["cat_map"][c]
            codes = df[c].map(mapping)
            codes = codes.where(~codes.isna(), -1).astype(float).to_numpy()
            cols.append(codes.reshape(-1, 1))
    return np.concatenate(cols, axis=1)


def run_test3b(train_panel_df, held_panel_df, gen_df, ind_df, numeric_cols, categorical_cols,
               n_boot=2000, boot_seed=424242, verifier_train_size=10000):
    panel_cols = numeric_cols + categorical_cols
    stats = fit_predictor_preprocessing(train_panel_df, numeric_cols, categorical_cols)

    rng = np.random.default_rng(REAL_VERIFIER_SUBSAMPLE_SEED)
    real_idx = rng.choice(len(train_panel_df), size=verifier_train_size, replace=False)
    real_df = train_panel_df.iloc[real_idx].reset_index(drop=True)

    per_target = {}
    rows_summary = []

    for j in panel_cols:
        predictors = [c for c in panel_cols if c != j]
        is_numeric = j in numeric_cols

        def fit_one(source_df):
            mask = source_df[j].notna().to_numpy()
            sub = source_df.loc[mask]
            X = build_design(sub, predictors, numeric_cols, categorical_cols, stats)
            y = sub[j].to_numpy(dtype=float) if is_numeric else sub[j].astype(str).to_numpy()
            if is_numeric:
                model = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3,
                                             max_features="sqrt",
                                             random_state=VERIFIER_RANDOM_STATE)
            else:
                model = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=3,
                                              max_features="sqrt",
                                              random_state=VERIFIER_RANDOM_STATE)
            model.fit(X, y)
            return model

        model_real = fit_one(real_df)
        model_gen = fit_one(gen_df)
        model_ind = fit_one(ind_df)

        eval_mask = held_panel_df[j].notna().to_numpy()
        eval_df = held_panel_df.loc[eval_mask]
        n_eval = len(eval_df)
        if n_eval == 0:
            continue
        X_eval = build_design(eval_df, predictors, numeric_cols, categorical_cols, stats)
        y_true = eval_df[j].to_numpy(dtype=float) if is_numeric else eval_df[j].astype(str).to_numpy()

        pred_real = model_real.predict(X_eval)
        pred_gen = model_gen.predict(X_eval)
        pred_ind = model_ind.predict(X_eval)

        if is_numeric:
            iqr = stats["num_iqr"][j]
            L_real = np.abs(pred_real - y_true) / iqr
            L_gen = np.abs(pred_gen - y_true) / iqr
            L_ind = np.abs(pred_ind - y_true) / iqr
        else:
            L_real = (pred_real != y_true).astype(float)
            L_gen = (pred_gen != y_true).astype(float)
            L_ind = (pred_ind != y_true).astype(float)

        Lj_real, Lj_gen, Lj_ind = L_real.mean(), L_gen.mean(), L_ind.mean()
        informative = Lj_real <= 0.95 * Lj_ind
        g_j = np.nan
        if informative and (Lj_ind - Lj_real) != 0:
            g_j = (Lj_ind - Lj_gen) / (Lj_ind - Lj_real)

        per_target[j] = {"L_real": L_real, "L_gen": L_gen, "L_ind": L_ind,
                          "informative": informative, "g_j": g_j}
        rows_summary.append({
            "coordinate": j, "type": "numeric" if is_numeric else "categorical",
            "n_eval": n_eval, "L_REAL": Lj_real, "L_GEN": Lj_gen, "L_IND": Lj_ind,
            "informative": bool(informative), "g_j": g_j,
        })

    by_target_df = pd.DataFrame(rows_summary)
    informative_targets = [j for j, v in per_target.items() if v["informative"]]
    if len(informative_targets) == 0:
        G_point = np.nan
    else:
        G_point = np.nanmean([per_target[j]["g_j"] for j in informative_targets])
    win_fraction = (np.mean([per_target[j]["g_j"] > 0 for j in informative_targets])
                    if informative_targets else np.nan)

    # bootstrap over held-out rows, per-target eligible sets, without refitting
    boot_master_rng = np.random.default_rng(boot_seed)
    G_boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        g_vals = []
        for j in informative_targets:
            d = per_target[j]
            n_j = len(d["L_real"])
            idx = boot_master_rng.integers(0, n_j, size=n_j)
            lr, lg, li = d["L_real"][idx].mean(), d["L_gen"][idx].mean(), d["L_ind"][idx].mean()
            denom = li - lr
            if denom != 0:
                g_vals.append((li - lg) / denom)
        G_boots[b] = np.mean(g_vals) if g_vals else np.nan

    G_lower = np.nanpercentile(G_boots, 2.5)
    G_upper = np.nanpercentile(G_boots, 97.5)

    return {
        "n_cross_structure_targets": len(informative_targets),
        "cross_structure_retention_G": float(G_point) if np.isfinite(G_point) else None,
        "cross_structure_retention_G_ci95_lower": float(G_lower) if np.isfinite(G_lower) else None,
        "cross_structure_retention_G_ci95_upper": float(G_upper) if np.isfinite(G_upper) else None,
        "cross_structure_target_win_fraction": float(win_fraction) if win_fraction == win_fraction else None,
        "by_target_df": by_target_df,
        "gate4_pass": bool(np.isfinite(G_lower) and G_lower > 0.70),
    }
