"""TEST 3B - cross-structure retention: fixed ExtraTrees verifiers trained on
real / generated / independent-marginal data, evaluated on the same
outer-held-out real materials.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

import config
from bootstrap import bootstrap_generic_ci

SEED_VERIFIER_REAL_TRAIN_SAMPLE = 707070
N_VERIFIER_TRAIN_ROWS = 10000


def _verifier(is_categorical: bool):
    kwargs = dict(n_estimators=100, min_samples_leaf=3, max_features="sqrt",
                  random_state=config.VERIFIER_RANDOM_STATE)
    return ExtraTreesClassifier(**kwargs) if is_categorical else ExtraTreesRegressor(**kwargs)


def build_encoders(train_frame: pd.DataFrame, numeric_panel, categorical_panel, quantizers):
    medians = {c: float(train_frame[c].median()) for c in numeric_panel}
    cat_maps = {c: quantizers[c] for c in categorical_panel}
    return medians, cat_maps


def encode_predictors(frame: pd.DataFrame, predictor_numeric, predictor_categorical, medians, cat_maps):
    cols = []
    names = []
    for c in predictor_numeric:
        v = frame[c].to_numpy(dtype=float).copy()
        v[np.isnan(v)] = medians[c]
        cols.append(v)
        names.append(c)
    for c in predictor_categorical:
        codes = cat_maps[c].transform(frame[c])
        cols.append(codes.astype(float))
        names.append(c)
    if not cols:
        return np.zeros((len(frame), 0))
    return np.column_stack(cols)


def run_test3b(train_frame, generated_frame, ind_frame, heldout_frame,
               numeric_panel, categorical_panel, quantizers, train_pos_frame):
    medians, cat_maps = build_encoders(train_pos_frame, numeric_panel, categorical_panel, quantizers)
    targets = numeric_panel + categorical_panel

    heldout_full = heldout_frame  # rows with ALL panel coords observed
    n_eval = len(heldout_full)

    per_target_rows = []
    loss_arrays = {}  # target -> dict(REAL/GEN/IND -> per-row loss array over heldout_full)

    rng = np.random.default_rng(SEED_VERIFIER_REAL_TRAIN_SAMPLE)

    for target in targets:
        is_cat = target in categorical_panel
        pred_numeric = [c for c in numeric_panel if c != target]
        pred_categorical = [c for c in categorical_panel if c != target]

        X_held = encode_predictors(heldout_full, pred_numeric, pred_categorical, medians, cat_maps)
        if is_cat:
            y_held = cat_maps[target].transform(heldout_full[target])
        else:
            y_held = heldout_full[target].to_numpy(dtype=float)

        real_pool = train_pos_frame[train_pos_frame[target].notna()] if not is_cat else \
            train_pos_frame[train_pos_frame[target] != config.MISSING_STATE]
        n_take = min(N_VERIFIER_TRAIN_ROWS, len(real_pool))
        real_sample = real_pool.sample(n=n_take, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))

        datasets = {"REAL": real_sample, "GEN": generated_frame, "IND": ind_frame}
        preds = {}
        for name, ds in datasets.items():
            Xd = encode_predictors(ds, pred_numeric, pred_categorical, medians, cat_maps)
            if is_cat:
                yd = cat_maps[target].transform(ds[target])
            else:
                yd = ds[target].to_numpy(dtype=float)
            model = _verifier(is_cat)
            model.fit(Xd, yd)
            preds[name] = model.predict(X_held)

        if is_cat:
            loss = {name: (preds[name] != y_held).astype(float) for name in datasets}
        else:
            train_full_vals = train_pos_frame[target].dropna().to_numpy(dtype=float)
            q75f, q25f = np.percentile(train_full_vals, [75, 25])
            iqr = q75f - q25f
            scale = iqr if iqr > 0 else (train_full_vals.max() - train_full_vals.min())
            scale = scale if scale > 0 else 1.0
            loss = {name: np.abs(preds[name] - y_held) / scale for name in datasets}

        loss_arrays[target] = loss

        L_REAL = float(loss["REAL"].mean())
        L_GEN = float(loss["GEN"].mean())
        L_IND = float(loss["IND"].mean())
        informative = L_REAL <= 0.95 * L_IND
        denom = L_IND - L_REAL
        g_j = (L_IND - L_GEN) / denom if abs(denom) > 1e-12 else np.nan

        per_target_rows.append({
            "target": target, "type": "categorical" if is_cat else "numeric",
            "L_REAL": L_REAL, "L_GEN": L_GEN, "L_IND": L_IND,
            "informative": bool(informative), "g_j": g_j,
            "gen_better_than_ind": bool(L_GEN < L_IND),
        })

    by_target_df = pd.DataFrame(per_target_rows)
    informative_targets = by_target_df.loc[by_target_df["informative"], "target"].tolist()

    def G_statistic(idx):
        vals = []
        for target in informative_targets:
            loss = loss_arrays[target]
            L_REAL = loss["REAL"][idx].mean()
            L_GEN = loss["GEN"][idx].mean()
            L_IND = loss["IND"][idx].mean()
            denom = L_IND - L_REAL
            if abs(denom) > 1e-12:
                vals.append((L_IND - L_GEN) / denom)
        return float(np.mean(vals)) if vals else np.nan

    rng2 = np.random.default_rng(config.SEED_VERIFIER_BOOTSTRAP)
    boot_vals = np.empty(config.N_VERIFIER_BOOTSTRAP)
    point = G_statistic(np.arange(n_eval))
    for b in range(config.N_VERIFIER_BOOTSTRAP):
        idx = rng2.integers(0, n_eval, size=n_eval)
        boot_vals[b] = G_statistic(idx)
    lo, hi = np.nanpercentile(boot_vals, [2.5, 97.5])

    win_fraction = float(by_target_df.loc[by_target_df["informative"], "gen_better_than_ind"].mean()) \
        if informative_targets else 0.0

    cross_structure_preserved = bool((lo > 0) and (win_fraction >= 0.60))

    summary = {
        "cross_structure_preserved": cross_structure_preserved,
        "n_cross_structure_targets": int(len(informative_targets)),
        "cross_structure_retention_G": point,
        "cross_structure_retention_G_ci95_lower": float(lo),
        "cross_structure_retention_G_ci95_upper": float(hi),
        "cross_structure_target_win_fraction": win_fraction,
        "n_heldout_eval_rows": int(n_eval),
    }
    return summary, by_target_df
