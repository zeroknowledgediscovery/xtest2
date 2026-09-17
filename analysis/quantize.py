"""Quantile-based discretization for numeric coordinates, with an explicit
missing state, plus discrete-state encoding for categorical coordinates.

The quantization resolution q is the single quantity permitted to be
selected by comparison on the inner fit/development split (see
config.QUANTIZATION_SELECTION_RULE). Everything else here is deterministic
given q and the fitting data.
"""
import numpy as np
import pandas as pd

import config


class NumericQuantizer:
    """One numeric coordinate: quantile bin edges fit on non-missing values
    of the fitting data, plus a dedicated missing bin (index = n_bins)."""

    def __init__(self, q: int):
        self.q = q
        self.edges = None       # interior bin edges, strictly increasing
        self.n_bins = None      # effective number of non-missing bins (<= q)
        self.bin_values = None  # list of np.ndarray: raw fit values per bin

    def fit(self, values: np.ndarray):
        obs = values[~np.isnan(values)]
        if obs.size == 0:
            self.edges = np.array([])
            self.n_bins = 1
            self.bin_values = [np.array([0.0])]
            return self
        quantile_points = np.linspace(0, 1, self.q + 1)[1:-1]
        edges = np.unique(np.quantile(obs, quantile_points))
        self.edges = edges
        self.n_bins = len(edges) + 1
        bin_idx = np.searchsorted(edges, obs, side="right")
        self.bin_values = [obs[bin_idx == k] for k in range(self.n_bins)]
        for k in range(self.n_bins):
            if self.bin_values[k].size == 0:
                self.bin_values[k] = obs
        return self

    @property
    def n_states(self):
        return self.n_bins + 1  # + missing state

    @property
    def missing_state(self):
        return self.n_bins

    def transform(self, values: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, self.missing_state, dtype=np.int64)
        mask = ~np.isnan(values)
        if mask.any():
            out[mask] = np.searchsorted(self.edges, values[mask], side="right")
        return out

    def sample_value(self, bin_index: int, rng: np.random.Generator, n: int = 1):
        pool = self.bin_values[bin_index]
        idx = rng.integers(0, len(pool), size=n)
        return pool[idx]


class CategoricalQuantizer:
    """One categorical coordinate: fixed level set discovered on the fitting
    data (including the explicit missing-state token as a level if present)."""

    def __init__(self):
        self.levels = None  # list of level strings, index = state

    def fit(self, values: pd.Series):
        levels = sorted(set(values.astype(str).tolist()))
        if config.MISSING_STATE in levels:
            levels.remove(config.MISSING_STATE)
            levels = levels + [config.MISSING_STATE]
        else:
            levels = levels + [config.MISSING_STATE]
        self.levels = levels
        self._index = {lv: i for i, lv in enumerate(levels)}
        return self

    @property
    def n_states(self):
        return len(self.levels)

    @property
    def missing_state(self):
        return self._index[config.MISSING_STATE]

    def transform(self, values: pd.Series) -> np.ndarray:
        return values.astype(str).map(lambda v: self._index.get(v, self.missing_state)).to_numpy(dtype=np.int64)

    def level_name(self, state: int) -> str:
        return self.levels[state]


def fit_quantizers(coord_frame: pd.DataFrame, fit_index, numeric_coords, categorical_coords, q: int):
    quantizers = {}
    fit_df = coord_frame.loc[fit_index]
    for col in numeric_coords:
        quantizers[col] = NumericQuantizer(q).fit(fit_df[col].to_numpy(dtype=float))
    for col in categorical_coords:
        quantizers[col] = CategoricalQuantizer().fit(fit_df[col])
    return quantizers


def discretize(coord_frame: pd.DataFrame, quantizers: dict, numeric_coords, categorical_coords) -> pd.DataFrame:
    out = pd.DataFrame(index=coord_frame.index)
    for col in numeric_coords:
        out[col] = quantizers[col].transform(coord_frame[col].to_numpy(dtype=float))
    for col in categorical_coords:
        out[col] = quantizers[col].transform(coord_frame[col])
    return out
