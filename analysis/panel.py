"""Fixed, mechanical generation-panel selection rule from instructions.md.
Applied only to the 50,000 outer-training rows; never touches held-out data."""
import json
import numpy as np

from . import config


def select_generation_panel(feat, train_pos):
    train = feat.iloc[train_pos]
    n = len(train)
    numeric_selected = []
    numeric_report = {}
    for c in config.MEASURED_NUMERIC_COLUMNS + config.CHEM_COLUMNS:
        v = train[c].to_numpy(dtype=float)
        finite = np.isfinite(v)
        coverage = finite.mean()
        n_distinct = len(np.unique(v[finite]))
        ok = (coverage >= config.GENERATION_MIN_COVERAGE) and (n_distinct >= config.GENERATION_MIN_DISTINCT)
        numeric_report[c] = {"coverage": float(coverage), "n_distinct": int(n_distinct), "included": bool(ok)}
        if ok:
            numeric_selected.append(c)

    categorical_selected = []
    categorical_report = {}
    for c in config.CATEGORICAL_COLUMNS:
        v = train[c]
        observed = v.notna()
        coverage = observed.mean()
        ok = coverage >= config.GENERATION_MIN_COVERAGE
        categorical_report[c] = {"coverage": float(coverage), "included": bool(ok)}
        if ok:
            categorical_selected.append(c)

    panel = {
        "numeric_columns": numeric_selected,
        "categorical_columns": categorical_selected,
        "selection_rule": (
            "numeric: finite training coverage >= 0.50 and >= 3 distinct observed "
            "values; categorical (crys, dimensionality): nonmissing training "
            "coverage >= 0.50. Evaluated only on the 50,000-row outer training set."
        ),
        "numeric_report": numeric_report,
        "categorical_report": categorical_report,
        "n_training_rows": int(n),
    }
    return panel


def write_generation_panel(panel, path):
    with open(path, "w") as f:
        json.dump(panel, f, indent=2, sort_keys=False)
