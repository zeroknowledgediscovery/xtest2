"""Deterministic chemistry summary features computed from `formula` only.

`formula` strings in this dataset are simple concatenations of
(ElementSymbol, optional integer count) tokens with no parentheses or
fractional counts (verified against the supplied jarvis_dft3d.pkl). Raw
`formula` itself is never used as a feature, embedding input, or lookup key;
only these twelve fixed numerical summaries derived from elemental
composition are used.
"""
import re
import numpy as np
from .periodic_table import ELEMENT_TABLE

_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

CHEM_FEATURE_NAMES = [
    "chem_n_elements",
    "chem_composition_entropy",
    "chem_mean_atomic_number",
    "chem_atomic_number_range",
    "chem_mean_electronegativity",
    "chem_electronegativity_range",
    "chem_mean_atomic_radius",
    "chem_atomic_radius_range",
    "chem_mean_group",
    "chem_group_range",
    "chem_mean_period",
    "chem_period_range",
]


def parse_formula(formula):
    """Return dict element symbol -> integer count."""
    counts = {}
    for sym, cnt in _TOKEN_RE.findall(formula):
        if not sym:
            continue
        n = int(cnt) if cnt else 1
        counts[sym] = counts.get(sym, 0) + n
    return counts


def chem_summary_row(formula):
    """Compute the 12 chemistry summary features for one formula string.

    Returns a dict of feature_name -> float, or all-NaN dict if the formula
    cannot be parsed into any recognized elements.
    """
    counts = parse_formula(formula)
    counts = {el: n for el, n in counts.items() if el in ELEMENT_TABLE}
    total = sum(counts.values())
    if total <= 0:
        return {k: np.nan for k in CHEM_FEATURE_NAMES}

    elems = list(counts.keys())
    fracs = np.array([counts[e] / total for e in elems], dtype=float)

    z = np.array([ELEMENT_TABLE[e]["Z"] for e in elems], dtype=float)
    en = np.array([ELEMENT_TABLE[e]["electronegativity"] for e in elems], dtype=float)
    rad = np.array([ELEMENT_TABLE[e]["atomic_radius"] for e in elems], dtype=float)
    grp = np.array([ELEMENT_TABLE[e]["group"] for e in elems], dtype=float)
    per = np.array([ELEMENT_TABLE[e]["period"] for e in elems], dtype=float)

    entropy = float(-np.sum(fracs * np.log(fracs)))

    def w_mean(vals):
        return float(np.sum(fracs * vals))

    def rng(vals):
        return float(np.max(vals) - np.min(vals))

    return {
        "chem_n_elements": float(len(elems)),
        "chem_composition_entropy": entropy,
        "chem_mean_atomic_number": w_mean(z),
        "chem_atomic_number_range": rng(z),
        "chem_mean_electronegativity": w_mean(en),
        "chem_electronegativity_range": rng(en),
        "chem_mean_atomic_radius": w_mean(rad),
        "chem_atomic_radius_range": rng(rad),
        "chem_mean_group": w_mean(grp),
        "chem_group_range": rng(grp),
        "chem_mean_period": w_mean(per),
        "chem_period_range": rng(per),
    }


def compute_chem_features(formula_series):
    """Vectorized-ish computation of chem summary dataframe from a formula Series."""
    import pandas as pd

    records = [chem_summary_row(f) for f in formula_series]
    return pd.DataFrame.from_records(records, index=formula_series.index)
