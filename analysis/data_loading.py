"""Data loading, canonical row identity, fixed outer/inner splits, and
construction of the allowed material-state coordinate frame.
"""
import os
import tarfile
import tempfile

import numpy as np
import pandas as pd

import config
from chemistry import chemistry_summary, CHEMISTRY_COLUMNS
from periodic_table_data import is_centrosymmetric


def load_raw_dataframe(data_path: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(data_path, "r:*") as tf:
            tf.extractall(tmp)
        pkl_path = None
        for root, _dirs, files in os.walk(tmp):
            for f in files:
                if f == "jarvis_dft3d.pkl":
                    pkl_path = os.path.join(root, f)
        if pkl_path is None:
            raise FileNotFoundError("jarvis_dft3d.pkl not found inside archive")
        df = pd.read_pickle(pkl_path)
    df = df.drop_duplicates(subset="jid", keep="first").reset_index(drop=True)
    return df


def _is_missing_scalar(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str):
        return v.strip().lower() in config.MISSING_TOKENS
    return False


def to_numeric_with_missing(series: pd.Series) -> pd.Series:
    """Coerce to float, mapping missing-like tokens (and unparsable values) to NaN."""
    cleaned = series.map(lambda v: np.nan if _is_missing_scalar(v) else v)
    return pd.to_numeric(cleaned, errors="coerce")


def to_categorical_with_missing(series: pd.Series) -> pd.Series:
    """Map missing-like tokens to the explicit missing-state token; otherwise
    keep the (stripped) string value."""
    def clean(v):
        if _is_missing_scalar(v):
            return config.MISSING_STATE
        if isinstance(v, str):
            return v.strip()
        return v
    return series.map(clean)


def compute_outer_split(df: pd.DataFrame):
    rng = np.random.default_rng(1)
    train_pos = rng.choice(len(df), size=50000, replace=False)
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_pos] = True
    train_jids = df.loc[train_pos, "jid"].to_numpy()
    test_jids = df.loc[~train_mask, "jid"].to_numpy()
    return train_pos, train_mask, train_jids, test_jids


def compute_inner_split(train_pos: np.ndarray):
    inner_rng = np.random.default_rng(271828)
    inner_dev_idx = inner_rng.choice(len(train_pos), size=10000, replace=False)
    inner_dev_mask = np.zeros(len(train_pos), dtype=bool)
    inner_dev_mask[inner_dev_idx] = True
    fit_pos = train_pos[~inner_dev_mask]
    inner_dev_pos = train_pos[inner_dev_mask]
    return fit_pos, inner_dev_pos


def build_coordinate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full allowed material-state coordinate frame (model input
    space): measured numeric columns, measured categorical columns, the 12
    deterministic chemistry summaries, and the deterministic symmetry
    descriptor. Row-aligned with df. jid is NOT included as a coordinate."""
    out = pd.DataFrame(index=df.index)

    for col in config.MEASURED_NUMERIC_COLUMNS:
        out[col] = to_numeric_with_missing(df[col]) if col in df.columns else np.nan

    for col in config.MEASURED_CATEGORICAL_COLUMNS:
        out[col] = to_categorical_with_missing(df[col]) if col in df.columns else config.MISSING_STATE

    chem = df["formula"].map(chemistry_summary).apply(pd.Series)
    for col in CHEMISTRY_COLUMNS:
        out[col] = chem[col] if col in chem.columns else np.nan

    def spg_centro(v):
        if _is_missing_scalar(v):
            return config.MISSING_STATE
        try:
            return str(is_centrosymmetric(int(str(v).strip())))
        except (ValueError, TypeError):
            return config.MISSING_STATE

    out["spg_centrosymmetric"] = df["spg_number"].map(spg_centro)

    return out


NUMERIC_COORDINATES = config.MEASURED_NUMERIC_COLUMNS + CHEMISTRY_COLUMNS
CATEGORICAL_COORDINATES = config.MEASURED_CATEGORICAL_COLUMNS + config.SYMMETRY_CATEGORICAL_COLUMNS
ALL_COORDINATES = NUMERIC_COORDINATES + CATEGORICAL_COORDINATES
