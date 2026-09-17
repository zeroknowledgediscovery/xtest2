"""Fixed generation panel selection (analysis/generation_panel.json)."""
import numpy as np
import pandas as pd

import config
from chemistry import CHEMISTRY_COLUMNS

MIN_COVERAGE = 0.50
MIN_DISTINCT = 3


def select_generation_panel(coord_frame: pd.DataFrame, train_pos: np.ndarray):
    sub = coord_frame.iloc[train_pos]
    numeric_candidates = config.MEASURED_NUMERIC_COLUMNS + CHEMISTRY_COLUMNS
    categorical_candidates = config.MEASURED_CATEGORICAL_COLUMNS

    numeric_panel, categorical_panel = [], []
    detail = {}
    for col in numeric_candidates:
        s = sub[col]
        coverage = float(s.notna().mean())
        distinct = int(s.dropna().nunique())
        included = coverage >= MIN_COVERAGE and distinct >= MIN_DISTINCT
        detail[col] = {"type": "numeric", "coverage": coverage, "distinct": distinct, "included": included}
        if included:
            numeric_panel.append(col)

    for col in categorical_candidates:
        s = sub[col]
        coverage = float((s != config.MISSING_STATE).mean())
        included = coverage >= MIN_COVERAGE
        detail[col] = {"type": "categorical", "coverage": coverage, "included": included}
        if included:
            categorical_panel.append(col)

    return numeric_panel, categorical_panel, detail
