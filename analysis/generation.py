"""TEST 3 (unconditional generation) and TEST 3A (marginal fidelity), plus
the independent-marginal negative-control baseline generator.
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import config


def decode_samples(discrete: np.ndarray, columns, quantizers, rng: np.random.Generator, numeric_coords, categorical_coords):
    out = {}
    col_index = {c: i for i, c in enumerate(columns)}
    for col in numeric_coords:
        q = quantizers[col]
        states = discrete[:, col_index[col]]
        vals = np.full(states.shape, np.nan)
        for k in range(q.n_bins):
            mask = states == k
            if mask.any():
                vals[mask] = q.sample_value(k, rng, n=int(mask.sum()))
        out[col] = vals
    for col in categorical_coords:
        q = quantizers[col]
        states = discrete[:, col_index[col]]
        out[col] = np.array([q.level_name(s) for s in states], dtype=object)
    return pd.DataFrame(out)


def independent_marginal_sample(train_frame: pd.DataFrame, numeric_cols, categorical_cols, n: int, seed: int = config.SEED_INDEPENDENT_MARGINAL):
    rng = np.random.default_rng(seed)
    out = {}
    for col in numeric_cols:
        obs = train_frame[col].dropna().to_numpy()
        idx = rng.integers(0, len(obs), size=n)
        out[col] = obs[idx]
    for col in categorical_cols:
        obs = train_frame[col][train_frame[col] != config.MISSING_STATE].to_numpy()
        idx = rng.integers(0, len(obs), size=n)
        out[col] = obs[idx]
    return pd.DataFrame(out)


def validity_and_uniqueness(samples: pd.DataFrame, numeric_panel, categorical_panel, train_frame: pd.DataFrame):
    n = len(samples)
    valid = np.ones(n, dtype=bool)
    for col in numeric_panel:
        valid &= np.isfinite(samples[col].to_numpy(dtype=float))
    for col in categorical_panel:
        valid &= (samples[col].to_numpy() != config.MISSING_STATE)

    def row_keys(df):
        cols = []
        for col in numeric_panel:
            v = df[col].to_numpy(dtype=float)
            formatted = np.where(np.isfinite(v), np.round(v, 8).astype(str), "NA")
            cols.append(formatted)
        for col in categorical_panel:
            v = df[col].to_numpy()
            formatted = np.where(v == config.MISSING_STATE, "NA", v.astype(str))
            cols.append(formatted)
        return list(zip(*cols))

    keys = row_keys(samples)
    n_unique = len(set(keys))

    train_keys = set(row_keys(train_frame.reset_index(drop=True)))
    n_exact_match = sum(1 for k, v in zip(keys, valid) if v and k in train_keys)

    return {
        "valid_count": int(valid.sum()),
        "requested_count": n,
        "unique_fraction": n_unique / n,
        "exact_training_match_fraction": n_exact_match / n,
        "valid_mask": valid,
    }


def marginal_fidelity(generated: pd.DataFrame, heldout: pd.DataFrame, numeric_panel, categorical_panel, train_frame: pd.DataFrame):
    rows = []
    for col in numeric_panel:
        gen_vals = generated[col].dropna().to_numpy()
        held_vals = heldout[col].dropna().to_numpy()
        if len(gen_vals) == 0 or len(held_vals) == 0:
            continue
        w1 = wasserstein_distance(gen_vals, held_vals)
        train_vals = train_frame[col].dropna().to_numpy()
        q75, q25 = np.percentile(train_vals, [75, 25])
        iqr = q75 - q25
        scale = iqr if iqr > 0 else (train_vals.max() - train_vals.min())
        scale = scale if scale > 0 else 1.0
        rows.append({"coordinate": col, "type": "numeric", "statistic": w1 / scale})
    for col in categorical_panel:
        gen_p = generated[col].value_counts(normalize=True)
        held_p = heldout[col].value_counts(normalize=True)
        cats = set(gen_p.index) | set(held_p.index)
        tv = 0.5 * sum(abs(gen_p.get(c, 0.0) - held_p.get(c, 0.0)) for c in cats)
        rows.append({"coordinate": col, "type": "categorical", "statistic": tv})
    return pd.DataFrame(rows)
