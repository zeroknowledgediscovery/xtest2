"""Anchor-bootstrap confidence interval helper shared by all tests."""
import numpy as np


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int, axis_size=None):
    """values: (A,) per-anchor statistic. Returns (mean, ci_lower, ci_upper)
    using n_boot resamples of the A anchors with replacement."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    a = values.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a, size=(n_boot, a))
    boot_means = values[idx].mean(axis=1)
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return float(values.mean()), float(lower), float(upper)


def bootstrap_generic_ci(per_unit_values: dict, statistic_fn, n_boot: int, seed: int, n_units: int):
    """per_unit_values: dict of arrays each of length n_units (aligned by unit
    index). statistic_fn(dict_of_resampled_arrays) -> scalar. Returns
    (point_estimate, lower, upper)."""
    rng = np.random.default_rng(seed)
    point = statistic_fn(per_unit_values)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        resampled = {k: v[idx] for k, v in per_unit_values.items()}
        boots[b] = statistic_fn(resampled)
    lower, upper = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lower), float(upper)
