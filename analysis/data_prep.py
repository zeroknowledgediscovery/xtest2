"""Data loading, cleaning, splitting, and the conventional reference
geometries d_E / d_M, exactly as specified in instructions.md."""
import os
import tarfile
import numpy as np
import pandas as pd

from . import config
from .chem_features import compute_chem_features


def _is_missing_token(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    s = str(v).strip().lower()
    return s in config.MISSING_TOKENS


def clean_numeric_series(s):
    """Treat missing-like tokens as missing, coerce everything else to float.
    Never silently replaces a numeric missing value with physical zero."""
    mask_missing = s.map(_is_missing_token)
    out = pd.to_numeric(s.where(~mask_missing), errors="coerce")
    return out


def clean_categorical_series(s):
    mask_missing = s.map(_is_missing_token)
    out = s.astype(object).where(~mask_missing, other=np.nan)
    out = out.map(lambda v: v.strip() if isinstance(v, str) else v)
    return out


def extract_dataframe(tgz_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tgz_path) as t:
        t.extract("jarvis_dft3d.pkl", path=extract_dir)
    df = pd.read_pickle(os.path.join(extract_dir, "jarvis_dft3d.pkl"))
    df = df.drop_duplicates(subset="jid", keep="first").reset_index(drop=True)
    return df


def build_feature_frame(df):
    """Return a cleaned frame containing measured columns (numeric + categorical),
    the 12 chemistry summaries, jid, and the 8 blind transport columns (left
    untouched, numeric-coerced, for later blind evaluation use only)."""
    out = pd.DataFrame(index=df.index)
    out["jid"] = df["jid"].to_numpy()

    for c in config.MEASURED_NUMERIC_COLUMNS:
        out[c] = clean_numeric_series(df[c]).to_numpy()
    for c in config.CATEGORICAL_COLUMNS:
        out[c] = clean_categorical_series(df[c]).to_numpy()

    chem = compute_chem_features(df["formula"])
    for c in config.CHEM_COLUMNS:
        out[c] = chem[c].to_numpy()

    # spg_number is used only, as explicitly permitted, as a nominal category
    # inside the conventional mixed-type comparator d_M -- never as an
    # ordinary continuous coordinate and never inside the structural model M.
    out["spg_number_cat"] = clean_categorical_series(df["spg_number"]).to_numpy()

    for c in config.BLIND_COLS:
        out[c] = clean_numeric_series(df[c]).to_numpy()

    return out


def make_outer_split(n_rows, jids):
    rng = np.random.default_rng(config.OUTER_SPLIT_SEED)
    train_pos = rng.choice(n_rows, size=config.OUTER_TRAIN_SIZE, replace=False)
    train_mask = np.zeros(n_rows, dtype=bool)
    train_mask[train_pos] = True
    train_jids = jids[train_pos]
    test_jids = jids[~train_mask]
    return train_pos, train_mask, train_jids, test_jids


def make_inner_split(train_pos):
    inner_rng = np.random.default_rng(config.INNER_DEV_SEED)
    inner_dev_idx = inner_rng.choice(len(train_pos), size=config.INNER_DEV_SIZE, replace=False)
    inner_dev_mask = np.zeros(len(train_pos), dtype=bool)
    inner_dev_mask[inner_dev_idx] = True
    fit_pos = train_pos[~inner_dev_mask]
    inner_dev_pos = train_pos[inner_dev_mask]
    return fit_pos, inner_dev_pos


# ---------------------------------------------------------------------------
# Conventional reference geometries d_E (numeric-only) and d_M (mixed-type)
# ---------------------------------------------------------------------------

def conventional_numeric_columns():
    return config.MEASURED_NUMERIC_COLUMNS + config.CHEM_COLUMNS


class ConventionalGeometry:
    """Standardized non-quantized Euclidean-type comparator d_E and mixed-type
    Gower-style comparator d_M, with standard deviations / ranges / category
    frequencies estimated only from the 50,000 outer-training rows."""

    def __init__(self, feat, train_pos):
        self.numeric_cols = conventional_numeric_columns()
        train = feat.iloc[train_pos]
        sigma = train[self.numeric_cols].std(ddof=1).to_numpy().copy()
        sigma[~np.isfinite(sigma) | (sigma == 0)] = np.nan
        self.sigma = sigma

        # Gower-style numeric range (max-min) from training data, plus
        # categorical columns used only by d_M.
        self.gower_range = (train[self.numeric_cols].max() - train[self.numeric_cols].min()).to_numpy().copy()
        self.gower_range[~np.isfinite(self.gower_range) | (self.gower_range == 0)] = np.nan

        self.cat_cols_M = ["crys", "dimensionality", "spg_number_cat"]

    def numeric_matrix(self, feat):
        return feat[self.numeric_cols].to_numpy(dtype=float)

    def d_E_pairwise(self, X):
        """X: (n, p) standardized-eligible raw numeric matrix (rows = eval cohort).
        Returns (n, n) matrix; np.nan where fewer than 10 jointly observed vars."""
        n, p = X.shape
        Z = (X - 0) / self.sigma  # sigma broadcasts; mean not needed, only scale
        obs = np.isfinite(Z)
        Zf = np.where(obs, Z, 0.0)
        # sum of squared standardized diffs over jointly observed coords:
        # (a-b)^2 = a^2 - 2ab + b^2, but only summed over jointly-observed dims.
        sq = Zf ** 2
        joint_obs = obs.astype(np.float64)
        n_joint = joint_obs @ joint_obs.T
        sum_a2 = sq @ joint_obs.T
        sum_b2 = joint_obs @ sq.T
        cross = Zf @ Zf.T
        sq_dist_sum = sum_a2 + sum_b2 - 2 * cross
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.sqrt(np.maximum(sq_dist_sum, 0.0) / n_joint)
        d[n_joint < 10] = np.nan
        np.fill_diagonal(d, 0.0)
        return d, n_joint

    def d_M_pairwise(self, feat_cohort):
        num_cols = self.numeric_cols
        X = feat_cohort[num_cols].to_numpy(dtype=float)
        n = X.shape[0]
        obs = np.isfinite(X)
        rng = self.gower_range
        Xn = np.where(obs, X / np.where(np.isfinite(rng), rng, np.nan), 0.0)
        rng_ok = np.isfinite(rng).astype(np.float64)
        obsf = obs.astype(np.float64) * rng_ok[None, :]

        sq = Xn ** 2
        n_joint_num = obsf @ obsf.T
        sum_a2 = sq @ obsf.T
        sum_b2 = obsf @ sq.T
        cross = Xn @ Xn.T
        num_sq_sum = np.maximum(sum_a2 + sum_b2 - 2 * cross, 0.0)

        spg_cat = feat_cohort["spg_number_cat"].to_numpy()
        cat_mats = []
        cat_obs_counts = np.zeros((n, n))
        cat_mismatch_sum = np.zeros((n, n))
        for col, vals in (
            ("crys", feat_cohort["crys"].to_numpy()),
            ("dimensionality", feat_cohort["dimensionality"].to_numpy()),
            ("spg_number_cat", spg_cat),
        ):
            is_obs = np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in vals])
            obs_pair = is_obs[:, None] & is_obs[None, :]
            eq = np.zeros((n, n), dtype=bool)
            vals_arr = np.array(vals, dtype=object)
            for i in range(n):
                if not is_obs[i]:
                    continue
                eq[i, :] = obs_pair[i, :] & (vals_arr == vals_arr[i])
            mismatch = obs_pair & (~eq)
            cat_obs_counts += obs_pair.astype(np.float64)
            cat_mismatch_sum += mismatch.astype(np.float64)

        total_joint = n_joint_num + cat_obs_counts
        total_sum = num_sq_sum + cat_mismatch_sum  # squared-diff for numeric, 0/1 for categorical
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.sqrt(total_sum / total_joint)
        d[total_joint < 10] = np.nan
        np.fill_diagonal(d, 0.0)
        return d
