"""Generic percentile bootstrap CI helper."""
import numpy as np


def bootstrap_ci(values, seed, n_boot=3000, stat_fn=None, alpha=0.05):
    """values: 1D array of per-unit (anchor or row) values.
    stat_fn: function(array)->scalar, default nanmean.
    Resamples unit indices with replacement, n_boot times, seed fixed.
    Returns (point_estimate, ci_lower, ci_upper, boot_stats array).
    """
    values = np.asarray(values, dtype=float)
    if stat_fn is None:
        stat_fn = lambda v: np.nanmean(v)
    n = len(values)
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot, dtype=float)
    idx_all = np.arange(n)
    for b in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        boot_stats[b] = stat_fn(values[idx])
    point = stat_fn(values)
    lower = np.nanpercentile(boot_stats, 100 * alpha / 2)
    upper = np.nanpercentile(boot_stats, 100 * (1 - alpha / 2))
    return point, lower, upper, boot_stats


def bootstrap_ci_indices(n_units, seed, n_boot, compute_stat_fn):
    """More general: compute_stat_fn(idx_array) -> scalar, resampling unit
    indices 0..n_units-1 with replacement. Returns (point, lower, upper, boots).
    Point estimate uses the identity resample (all indices once).
    """
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    idx_all = np.arange(n_units)
    point = compute_stat_fn(idx_all)
    for b in range(n_boot):
        idx = rng.choice(idx_all, size=n_units, replace=True)
        boots[b] = compute_stat_fn(idx)
    lower = np.nanpercentile(boots, 2.5)
    upper = np.nanpercentile(boots, 97.5)
    return point, lower, upper, boots
