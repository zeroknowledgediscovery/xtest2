"""Fixed negative-control generator: samples each coordinate independently
from its own empirical training marginal (the 50,000 outer-training rows),
per instructions.md. Uses the same fitted per-column marginals as the
structural model (no separate fitting), but discards all joint/copula
structure -- exactly the required negative control."""
import numpy as np

from . import config


class IndependentMarginalBaseline:
    def __init__(self, model):
        self.model = model
        self.columns = model.columns
        self.numeric_cols = model.numeric_cols
        self.categorical_cols = model.categorical_cols

    def sample_unconditional(self, n, rng):
        out = {}
        for c in self.numeric_cols:
            pool = self.model.marginals[c].sorted_train
            out[c] = rng.choice(pool, size=n, replace=True)
        for c in self.categorical_cols:
            m = self.model.marginals[c]
            probs = np.diff(m.cum_edges)
            idx = rng.choice(len(m.categories), size=n, replace=True, p=probs)
            out[c] = np.array([m.categories[i] for i in idx], dtype=object)
        return out

    def sample_conditional(self, observed_dict, hidden_mask_per_col, n, rng):
        """observed_dict: {col: array of length n raw values, with np.nan/None
        where hidden}. hidden_mask_per_col: {col: bool array length n, True
        where hidden}. Returns dict col -> array length n with hidden entries
        resampled independently from the training marginal, observed entries
        left untouched."""
        out = {}
        for c in self.columns:
            vals = np.array(observed_dict[c], dtype=object)
            hidden = hidden_mask_per_col[c]
            n_hidden = int(hidden.sum())
            if n_hidden == 0:
                out[c] = vals
                continue
            if c in self.numeric_cols:
                pool = self.model.marginals[c].sorted_train
                draws = rng.choice(pool, size=n_hidden, replace=True)
            else:
                m = self.model.marginals[c]
                probs = np.diff(m.cum_edges)
                idx = rng.choice(len(m.categories), size=n_hidden, replace=True, p=probs)
                draws = np.array([m.categories[i] for i in idx], dtype=object)
            vals = vals.copy()
            vals[hidden] = draws
            out[c] = vals
        return out
