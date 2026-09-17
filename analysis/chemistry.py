"""Deterministic chemistry summary features derived only from `formula`.

Exactly the 12 permitted summaries. No fitted parameters; pure functions of
the chemical formula string and the fixed periodic-table constants in
periodic_table_data.py.
"""
import re
import numpy as np

from periodic_table_data import ATOMIC_NUMBER, ELECTRONEGATIVITY, ATOMIC_RADIUS, GROUP, PERIOD

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")

CHEMISTRY_COLUMNS = [
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


def parse_formula(formula: str):
    """Return dict element_symbol -> atom count for a formula string."""
    counts = {}
    if not isinstance(formula, str) or not formula:
        return counts
    for sym, cnt in _FORMULA_TOKEN.findall(formula):
        if not sym:
            continue
        if sym not in ATOMIC_NUMBER:
            continue
        n = float(cnt) if cnt not in ("", None) else 1.0
        if n == 0.0:
            n = 1.0
        counts[sym] = counts.get(sym, 0.0) + n
    return counts


def chemistry_summary(formula: str):
    """Return a dict with the 12 chemistry summary features, or all-NaN if
    the formula cannot be parsed into any known element."""
    counts = parse_formula(formula)
    if not counts:
        return {c: np.nan for c in CHEMISTRY_COLUMNS}

    elems = list(counts.keys())
    total = sum(counts.values())
    fracs = np.array([counts[e] / total for e in elems], dtype=float)

    z = np.array([ATOMIC_NUMBER[e] for e in elems], dtype=float)
    en = np.array([ELECTRONEGATIVITY[e] for e in elems], dtype=float)
    radius = np.array([ATOMIC_RADIUS[e] for e in elems], dtype=float)
    group = np.array([GROUP[e] for e in elems], dtype=float)
    period = np.array([PERIOD[e] for e in elems], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = float(-np.sum(fracs * np.log(fracs)))

    def wmean(x):
        return float(np.sum(fracs * x))

    def rng(x):
        return float(np.max(x) - np.min(x))

    return {
        "chem_n_elements": float(len(elems)),
        "chem_composition_entropy": entropy,
        "chem_mean_atomic_number": wmean(z),
        "chem_atomic_number_range": rng(z),
        "chem_mean_electronegativity": wmean(en),
        "chem_electronegativity_range": rng(en),
        "chem_mean_atomic_radius": wmean(radius),
        "chem_atomic_radius_range": rng(radius),
        "chem_mean_group": wmean(group),
        "chem_group_range": rng(group),
        "chem_mean_period": wmean(period),
        "chem_period_range": rng(period),
    }
