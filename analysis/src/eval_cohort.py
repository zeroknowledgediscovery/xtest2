"""Fixed 3,000-material geometry/retrieval evaluation cohort, shared by
TEST1 (geometry novelty) and TEST2 (blind distant-analog retrieval)."""
import numpy as np
import pandas as pd

from .data_prep import BLIND_COLS, to_numeric_with_missing


def build_eval_cohort(df, train_mask):
    blind_numeric = pd.DataFrame({c: to_numeric_with_missing(df[c]) for c in BLIND_COLS})
    eligible_pos = np.flatnonzero((~train_mask) & blind_numeric.notna().all(axis=1).to_numpy())
    eval_rng = np.random.default_rng(20260915)
    eval_pos = eval_rng.choice(eligible_pos, size=3000, replace=False)
    return eval_pos, blind_numeric


def standardize_blind(blind_numeric, train_mask):
    train_vals = blind_numeric.loc[train_mask]
    mean = train_vals.mean(axis=0)
    sd = train_vals.std(axis=0, ddof=1)
    return mean, sd


def pairwise_d_F(blind_cohort, mean, sd):
    Z = (blind_cohort.to_numpy(dtype=float) - mean.to_numpy()[None, :]) / sd.to_numpy()[None, :]
    n = Z.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diff = Z[i][None, :] - Z
        D[i, :] = np.sqrt(np.mean(diff * diff, axis=1))
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    return D
