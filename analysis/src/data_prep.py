"""Authoritative dataframe construction, fixed outer split, missing-token
normalization, and allowed feature-frame assembly for the JARVIS dft_3d
structural-model task.
"""
import tarfile
import io
import json
import numpy as np
import pandas as pd

from .chem_features import compute_chem_features, CHEM_FEATURE_NAMES

MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "--", "missing"}

# Allowed measured-property columns (may be present or absent; all present here).
ALLOWED_NUMERIC_COLS = [
    "nat",
    "density",
    "formation_energy_peratom",
    "ehull",
    "exfoliation_energy",
    "optb88vdw_bandgap",
    "mbj_bandgap",
    "hse_gap",
    "spillage",
    "magmom_oszicar",
    "magmom_outcar",
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "avg_elec_mass",
    "avg_hole_mass",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "poisson",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_dielectric_electronic",
    "dfpt_piezo_max_dielectric_ionic",
    "max_ir_mode",
    "min_ir_mode",
    "Tc_supercon",
]

ALLOWED_CATEGORICAL_COLS = ["crys", "dimensionality"]
SPG_COL = "spg_number"

BLIND_COLS = [
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "ncond",
    "pcond",
    "nkappa",
    "pkappa",
]


def load_raw(tgz_path):
    with tarfile.open(tgz_path, "r:*") as tf:
        member = None
        for m in tf.getmembers():
            if m.name.endswith("jarvis_dft3d.pkl"):
                member = m
                break
        if member is None:
            raise FileNotFoundError("jarvis_dft3d.pkl not found inside tgz")
        f = tf.extractfile(member)
        buf = io.BytesIO(f.read())
    return pd.read_pickle(buf)


def build_authoritative_df(raw):
    df = raw.drop_duplicates(subset="jid", keep="first").reset_index(drop=True)
    return df


def compute_outer_split(df):
    rng = np.random.default_rng(1)
    train_pos = rng.choice(len(df), size=50000, replace=False)
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_pos] = True
    train_jids = df.loc[train_pos, "jid"].to_numpy()
    test_jids = df.loc[~train_mask, "jid"].to_numpy()
    return train_pos, train_mask, train_jids, test_jids


def _is_missing_token(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    s = str(v).strip().lower()
    return s in MISSING_TOKENS


def to_numeric_with_missing(series):
    """Coerce a (possibly object/mixed) column to float, treating missing-like
    tokens as NaN. Never substitutes physical zero for missing."""
    cleaned = series.map(lambda v: np.nan if _is_missing_token(v) else v)
    return pd.to_numeric(cleaned, errors="coerce")


def to_categorical_with_missing(series):
    cleaned = series.map(lambda v: np.nan if _is_missing_token(v) else str(v))
    return cleaned


def build_feature_frame(df):
    """Build the full allowed material-state feature frame:
    numeric measured-property columns + 12 chemistry summaries (numeric),
    plus crys/dimensionality/spg_number as categorical columns.
    Index-aligned with df (0..N-1 positions).
    """
    out = {}
    for col in ALLOWED_NUMERIC_COLS:
        out[col] = to_numeric_with_missing(df[col])
    numeric_frame = pd.DataFrame(out, index=df.index)

    chem = compute_chem_features(df["formula"])

    cat = {}
    for col in ALLOWED_CATEGORICAL_COLS:
        cat[col] = to_categorical_with_missing(df[col])
    cat[SPG_COL] = to_categorical_with_missing(df[SPG_COL])
    cat_frame = pd.DataFrame(cat, index=df.index)

    feat = pd.concat([numeric_frame, chem, cat_frame], axis=1)
    return feat


def get_blind_frame(df):
    out = {}
    for col in BLIND_COLS:
        out[col] = to_numeric_with_missing(df[col])
    return pd.DataFrame(out, index=df.index)


def select_generation_panel(feat, train_mask):
    """Select generation panel coordinates using the fixed coverage rule,
    computed from the 50,000 training rows only.

    Numeric coordinates (measured-property numeric + 12 chem summaries):
        finite coverage >= 0.50 AND distinct observed values >= 3
    Categorical coordinates (crys, dimensionality only):
        nonmissing coverage >= 0.50
    Returns dict with 'numeric' and 'categorical' coordinate name lists,
    in a fixed deterministic order.
    """
    train_feat = feat.loc[train_mask]
    n_train = len(train_feat)

    numeric_candidates = ALLOWED_NUMERIC_COLS + CHEM_FEATURE_NAMES
    numeric_panel = []
    for col in numeric_candidates:
        s = train_feat[col]
        finite = s.notna()
        coverage = finite.sum() / n_train
        distinct = s[finite].nunique()
        if coverage >= 0.50 and distinct >= 3:
            numeric_panel.append(col)

    categorical_panel = []
    for col in ALLOWED_CATEGORICAL_COLS:
        s = train_feat[col]
        coverage = s.notna().sum() / n_train
        if coverage >= 0.50:
            categorical_panel.append(col)

    return {"numeric": numeric_panel, "categorical": categorical_panel}
