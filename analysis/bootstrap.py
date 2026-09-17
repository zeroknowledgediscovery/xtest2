"""Generic vectorized anchor-bootstrap utility."""
import numpy as np


def bootstrap_ci(n_anchors, seed, n_boot, value_arrays, stat_fn, alpha=0.05):
    """value_arrays: tuple of 1D numpy arrays, each length n_anchors (anchor-
    indexed statistics already computed once). stat_fn(*arrays_2d) must
    operate along axis=-1 of 2D inputs of shape (k, n_anchors) and return a
    1D array of length k.

    Returns (point_estimate, ci_lower, ci_upper, boot_distribution)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_anchors, size=(n_boot, n_anchors))
    resampled = [a[idx] for a in value_arrays]
    boot_vals = stat_fn(*resampled)
    point_inputs = [a[None, :] for a in value_arrays]
    point = stat_fn(*point_inputs)[0]
    lower = np.percentile(boot_vals, 100 * alpha / 2)
    upper = np.percentile(boot_vals, 100 * (1 - alpha / 2))
    return float(point), float(lower), float(upper), boot_vals


def mean_stat(a):
    return np.mean(a, axis=-1)


def ratio_of_means_stat(num, den):
    return np.mean(num, axis=-1) / np.mean(den, axis=-1)
