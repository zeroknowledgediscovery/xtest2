"""TEST 3 -- unconditional generation validity."""
import numpy as np
import pandas as pd


def round_for_dedup(df, numeric_cols, categorical_cols, decimals=8):
    out = df.copy()
    for c in numeric_cols:
        out[c] = np.round(out[c].astype(float), decimals)
    for c in categorical_cols:
        out[c] = out[c].astype(str)
    return out


def check_validity(samples, numeric_cols, categorical_cols, num_global_min, num_global_max,
                    cat_categories):
    n = len(samples)
    valid = np.ones(n, dtype=bool)
    for j, c in enumerate(numeric_cols):
        v = samples[c].to_numpy(dtype=float)
        finite = np.isfinite(v)
        within = finite & (v >= num_global_min[j] - 1e-6) & (v <= num_global_max[j] + 1e-6)
        valid &= within
    for c in categorical_cols:
        allowed = set(cat_categories[c])
        valid &= samples[c].isin(allowed).to_numpy()
    return valid


def uniqueness_and_matches(samples, train_panel_df, numeric_cols, categorical_cols):
    rounded = round_for_dedup(samples, numeric_cols, categorical_cols)
    panel_cols = numeric_cols + categorical_cols
    tuples = list(map(tuple, rounded[panel_cols].to_numpy()))
    n = len(tuples)
    unique_fraction = len(set(tuples)) / n

    complete_train = train_panel_df[panel_cols].dropna()
    complete_rounded = round_for_dedup(complete_train, numeric_cols, categorical_cols)
    train_set = set(map(tuple, complete_rounded[panel_cols].to_numpy()))

    exact_matches = sum(1 for t in tuples if t in train_set)
    exact_fraction = exact_matches / n
    return unique_fraction, exact_fraction


def marginal_fidelity(samples, train_panel_df, numeric_cols, categorical_cols):
    rows = []
    for c in numeric_cols:
        gen = samples[c].to_numpy(dtype=float)
        real = train_panel_df[c].dropna().to_numpy(dtype=float)
        rows.append({
            "coordinate": c, "type": "numeric",
            "gen_mean": float(np.mean(gen)), "real_mean": float(np.mean(real)),
            "gen_sd": float(np.std(gen, ddof=1)), "real_sd": float(np.std(real, ddof=1)),
        })
    for c in categorical_cols:
        gen_counts = samples[c].value_counts(normalize=True)
        real_counts = train_panel_df[c].dropna().value_counts(normalize=True)
        cats = sorted(set(gen_counts.index) | set(real_counts.index))
        tv = 0.5 * sum(abs(gen_counts.get(k, 0.0) - real_counts.get(k, 0.0)) for k in cats)
        rows.append({
            "coordinate": c, "type": "categorical",
            "gen_mean": np.nan, "real_mean": np.nan,
            "gen_sd": np.nan, "real_sd": np.nan, "total_variation_distance": tv,
        })
    return pd.DataFrame(rows)
