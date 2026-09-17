"""
Deterministic chemistry summaries derived only from `formula`, exactly as
specified in instructions.md. No fitted parameters. Raw `formula` strings are
never retained as features, identity tokens, or lookup keys -- only the 12
numeric summaries below are exposed.
"""
import re
import numpy as np
import pandas as pd

from .periodic_table import ATOMIC_NUMBER, PERIOD, GROUP, ELECTRONEGATIVITY, ATOMIC_RADIUS

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")

CHEM_COLUMNS = [
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
    """Return dict {element_symbol: atom_count} or {} if unparseable."""
    if not isinstance(formula, str) or formula == "":
        return {}
    counts = {}
    total_chars = 0
    for elem, num in _FORMULA_TOKEN.findall(formula):
        if not elem:
            continue
        if elem not in ATOMIC_NUMBER:
            return {}
        n = float(num) if num else 1.0
        counts[elem] = counts.get(elem, 0.0) + n
        total_chars += len(elem) + len(num)
    if total_chars != len(formula) or not counts:
        return {}
    return counts


def chem_features_for_formula(formula):
    counts = parse_formula(formula)
    if not counts:
        return {c: np.nan for c in CHEM_COLUMNS}
    elems = list(counts.keys())
    n_atoms = sum(counts.values())
    fractions = {e: counts[e] / n_atoms for e in elems}

    def wmean(table):
        return sum(fractions[e] * table[e] for e in elems)

    def rng(table):
        vals = [table[e] for e in elems]
        return max(vals) - min(vals)

    entropy = -sum(x * np.log(x) for x in fractions.values() if x > 0)

    return {
        "chem_n_elements": float(len(elems)),
        "chem_composition_entropy": float(entropy),
        "chem_mean_atomic_number": float(wmean(ATOMIC_NUMBER)),
        "chem_atomic_number_range": float(rng(ATOMIC_NUMBER)),
        "chem_mean_electronegativity": float(wmean(ELECTRONEGATIVITY)),
        "chem_electronegativity_range": float(rng(ELECTRONEGATIVITY)),
        "chem_mean_atomic_radius": float(wmean(ATOMIC_RADIUS)),
        "chem_atomic_radius_range": float(rng(ATOMIC_RADIUS)),
        "chem_mean_group": float(wmean(GROUP)),
        "chem_group_range": float(rng(GROUP)),
        "chem_mean_period": float(wmean(PERIOD)),
        "chem_period_range": float(rng(PERIOD)),
    }


def compute_chem_features(formula_series):
    records = [chem_features_for_formula(f) for f in formula_series]
    return pd.DataFrame(records, index=formula_series.index, columns=CHEM_COLUMNS)
