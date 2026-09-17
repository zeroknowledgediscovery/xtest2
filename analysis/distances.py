"""Conventional (non-quantized) reference geometries used only to test
whether the submitted geometry d_G is novel: standardized Euclidean d_E and
a Gower-style mixed-type comparator d_M. Neither is part of the submitted
structural model.
"""
import numpy as np
import pandas as pd

MIN_JOINT_NUMERIC = 10


def _pairwise_sq_and_count(Z: np.ndarray, M: np.ndarray):
    """Z: (n, d) standardized values with 0 where missing. M: (n, d) 0/1
    observed mask. Returns (numerator (n,n) sum of squared diffs over
    jointly-observed coords, denom (n,n) joint-observed count)."""
    V = M * Z
    term2 = V @ V.T
    ZM2 = M * (Z ** 2)
    term_row = ZM2 @ M.T  # sum_j M[a,j] Z[a,j]^2 M[b,j]
    numerator = term_row + term_row.T - 2 * term2
    denom = M @ M.T
    return numerator, denom


def standardized_euclidean_distance(values: pd.DataFrame, sigma: pd.Series):
    cols = values.columns
    X = values.to_numpy(dtype=float)
    M = (~np.isnan(X)).astype(float)
    sig = sigma.reindex(cols).to_numpy(dtype=float)
    sig_safe = np.where((sig <= 0) | ~np.isfinite(sig), 1.0, sig)
    Z = np.where(M > 0, X / sig_safe[None, :], 0.0)
    numerator, denom = _pairwise_sq_and_count(Z, M)
    eligible = denom >= MIN_JOINT_NUMERIC
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.sqrt(np.clip(numerator, 0, None) / np.clip(denom, 1, None))
    np.fill_diagonal(d, 0.0)
    return d, eligible, denom


def gower_mixed_distance(numeric_values: pd.DataFrame, numeric_range: pd.Series,
                          categorical_values: pd.DataFrame):
    num_cols = numeric_values.columns
    X = numeric_values.to_numpy(dtype=float)
    M = (~np.isnan(X)).astype(float)
    rng = numeric_range.reindex(num_cols).to_numpy(dtype=float)
    rng_safe = np.where((rng <= 0) | ~np.isfinite(rng), 1.0, rng)
    Xs = np.where(M > 0, X / rng_safe[None, :], 0.0)
    # |xa/r - xb/r| via same matrix trick, but abs not square; use squared trick
    # then take sqrt is wrong for abs diff of already-scaled values; compute directly.
    n = X.shape[0]
    numerator = np.zeros((n, n))
    denom = np.zeros((n, n))
    Xs_masked = np.where(M > 0, Xs, np.nan)
    for j in range(X.shape[1]):
        col = Xs_masked[:, j]
        obs = M[:, j]
        both = obs[:, None] * obs[None, :]
        diff = np.abs(col[:, None] - col[None, :])
        diff = np.where(np.isnan(diff), 0.0, diff)
        numerator += both * diff
        denom += both

    for col in categorical_values.columns:
        vals = categorical_values[col].to_numpy()
        is_missing = pd.isna(vals) | (vals == "__MISSING__")
        both = (~is_missing[:, None]) & (~is_missing[None, :])
        mismatch = (vals[:, None] != vals[None, :]).astype(float)
        numerator += both * mismatch
        denom += both.astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        d = numerator / np.clip(denom, 1, None)
    np.fill_diagonal(d, 0.0)
    eligible = denom >= 1
    return d, eligible, denom
