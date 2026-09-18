"""Independent-marginal baseline P_IND(X) = prod_j P_train(X_j).

Used both as a generative baseline (TEST3B "IND" verifier training set) and
for conditional completion (TEST4 baseline): hidden coordinates are sampled
independently from their own training marginal, ignoring all other
coordinates.
"""
import numpy as np
import pandas as pd


class IndependentMarginalModel:
    def __init__(self, panel_df, numeric_cols, categorical_cols):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.panel_cols = self.numeric_cols + self.categorical_cols
        self.num_values = {}
        for col in self.numeric_cols:
            v = panel_df[col].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            self.num_values[col] = v
        self.cat_values = {}
        for col in self.categorical_cols:
            v = panel_df[col].dropna().to_numpy()
            self.cat_values[col] = v

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        out = {}
        for col in self.numeric_cols:
            vals = self.num_values[col]
            out[col] = rng.choice(vals, size=n, replace=True)
        for col in self.categorical_cols:
            vals = self.cat_values[col]
            out[col] = rng.choice(vals, size=n, replace=True)
        return pd.DataFrame(out, columns=self.panel_cols)

    def sample_column(self, col, n, rng):
        if col in self.num_values:
            vals = self.num_values[col]
        else:
            vals = self.cat_values[col]
        return rng.choice(vals, size=n, replace=True)
