"""Standardized non-quantized Euclidean comparator d_E and mixed Gower-style
comparator d_M, computed over a fixed evaluation cohort. All scale/range
statistics are estimated from the 50,000 training rows only.
"""
import numpy as np

from .data_prep import ALLOWED_NUMERIC_COLS
from .chem_features import CHEM_FEATURE_NAMES

D_E_NUMERIC_COLS = ALLOWED_NUMERIC_COLS + CHEM_FEATURE_NAMES
D_M_CATEGORICAL_COLS = ["crys", "dimensionality", "spg_number"]

MIN_JOINT = 10


def fit_distance_stats(feat, train_mask):
    train = feat.loc[train_mask]
    sigma = {}
    for col in D_E_NUMERIC_COLS:
        s = train[col].astype(float)
        sd = np.nanstd(s.to_numpy(), ddof=1)
        sigma[col] = sd if sd > 0 and np.isfinite(sd) else np.nan

    ranges = {}
    for col in D_E_NUMERIC_COLS:
        s = train[col].astype(float).to_numpy()
        if np.all(np.isnan(s)):
            ranges[col] = np.nan
            continue
        r = np.nanmax(s) - np.nanmin(s)
        ranges[col] = r if r > 0 else np.nan

    return {"sigma": sigma, "ranges": ranges}


def _standardized_matrix(feat_cohort, sigma):
    cols = D_E_NUMERIC_COLS
    X = feat_cohort[cols].astype(float).to_numpy()
    sig = np.array([sigma[c] for c in cols])
    valid_col = np.isfinite(sig)
    Xs = np.full_like(X, np.nan)
    Xs[:, valid_col] = X[:, valid_col] / sig[valid_col]
    return Xs


def pairwise_d_E(feat_cohort, sigma, min_joint=MIN_JOINT):
    """Return (n,n) symmetric distance matrix, np.inf where <min_joint jointly
    observed coordinates."""
    Xs = _standardized_matrix(feat_cohort, sigma)
    n, p = Xs.shape
    mask = ~np.isnan(Xs)
    Xf = np.where(mask, Xs, 0.0)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diff = Xf[i][None, :] - Xf
        joint = mask[i][None, :] & mask
        diff = diff * joint
        sumsq = np.einsum("ij,ij->i", diff, diff)
        count = joint.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.sqrt(sumsq / count)
        d[count < min_joint] = np.inf
        D[i, :] = d
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    return D


def pairwise_d_M(feat_cohort, ranges, min_joint=MIN_JOINT):
    """Gower-style mixed distance: numeric via range-normalized abs diff,
    categorical via mismatch indicator, averaged over jointly observed
    variables (both numeric and categorical)."""
    num_cols = D_E_NUMERIC_COLS
    Xn = feat_cohort[num_cols].astype(float).to_numpy()
    rng = np.array([ranges[c] for c in num_cols])
    valid_col = np.isfinite(rng)
    n = Xn.shape[0]

    cat_cols = D_M_CATEGORICAL_COLS
    Xc = feat_cohort[cat_cols].to_numpy(dtype=object)
    cat_mask = ~pd_isna_object(Xc)

    D = np.zeros((n, n), dtype=np.float64)
    num_mask = ~np.isnan(Xn)

    for i in range(n):
        # numeric contribution
        diff = np.abs(Xn[i][None, :] - Xn)
        with np.errstate(invalid="ignore"):
            diff_scaled = np.where(valid_col[None, :], diff / rng[None, :], np.nan)
        joint_num = num_mask[i][None, :] & num_mask & valid_col[None, :]
        num_sum = np.where(joint_num, np.nan_to_num(diff_scaled, nan=0.0), 0.0).sum(axis=1)
        num_count = joint_num.sum(axis=1)

        # categorical contribution
        joint_cat = cat_mask[i][None, :] & cat_mask
        mismatch = (Xc[i][None, :] != Xc).astype(float)
        cat_sum = np.where(joint_cat, mismatch, 0.0).sum(axis=1)
        cat_count = joint_cat.sum(axis=1)

        total_count = num_count + cat_count
        total_sum = num_sum + cat_sum
        with np.errstate(invalid="ignore", divide="ignore"):
            d = total_sum / total_count
        d[total_count < min_joint] = np.inf
        D[i, :] = d
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    return D


def pd_isna_object(arr):
    import pandas as pd
    return pd.isna(arr)
