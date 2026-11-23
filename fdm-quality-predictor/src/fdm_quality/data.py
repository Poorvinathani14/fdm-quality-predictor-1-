"""Data loading and cleaning for the FDM 3D-printing dataset (Mendeley Data, DOI 10.17632/zd6td6svd6.2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = {"Tensile strength (MPa)": "tensile", "Surface roughness (µm)": "roughness"}
LABEL = {
    "printer": "Printer type", "material": "Material", "pattern": "Infill pattern",
    "layer": "Layer thickness", "infill": "Infill density", "speed": "Print speed",
    "width": "Specimen width", "spec_t": "Specimen thickness",
}


def read_table(path: str) -> pd.DataFrame:
    """Read the dataset from .csv or .xlsx (picks the sheet that looks like the main table)."""
    if path.lower().endswith(".csv"):
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    sheets = pd.read_excel(path, sheet_name=None)
    for name, d in sheets.items():
        cols = " ".join(map(str, d.columns)).lower()
        if "material" in cols and "rough" in cols:
            return d
    return sheets[list(sheets)[0]]


def _find(raw: pd.DataFrame, *must, avoid=(), required=True):
    for c in raw.columns:
        low = c.lower()
        if all(m in low for m in must) and not any(a in low for a in avoid):
            return c
    if required:
        raise KeyError(f"Could not find a column containing {must}. Columns: {list(raw.columns)}")
    return None


def map_columns(raw: pd.DataFrame) -> dict:
    """Map the file's column names to short internal names by keyword."""
    return {
        "printer": _find(raw, "printer", required=False),
        "material": _find(raw, "material"),
        "layer": _find(raw, "layer"),
        "infill": _find(raw, "infill", "dens"),
        "pattern": _find(raw, "pattern"),
        "speed": _find(raw, "speed"),
        "width": _find(raw, "width", required=False),
        "spec_t": _find(raw, "thick", avoid=("layer",), required=False),
        "stress": _find(raw, "stress"),
        "rough": _find(raw, "rough"),
        "load": _find(raw, "load", required=False),
    }


def _clean_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace("\u200b", "", regex=False).str.strip().str.lower()
    return s.str.replace(r"\s+", " ", regex=True)


def prepare(raw: pd.DataFrame, use_specimen_dims: bool = False, drop_zero_roughness: bool = True,
            verbose: bool = True):
    """Clean the raw table.

    Returns (df, groups, config) where config holds FEATURES / CAT / NUM / GROUP_COLS.
    """
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    col = map_columns(raw)
    if verbose:
        print("COLUMN MAPPING (check it!)")
        for k, v in col.items():
            print(f"  {k:9s} <- {v}")

    df = pd.DataFrame()
    for key in ["printer", "material", "pattern"]:
        if col[key]:
            df[key] = _clean_text(raw[col[key]])
    for key in ["layer", "infill", "speed", "width", "spec_t", "stress", "rough", "load"]:
        if col.get(key):
            df[key] = pd.to_numeric(raw[col[key]], errors="coerce")

    sname = col["stress"].lower()
    df["tensile"] = df["stress"] if "mpa" in sname else df["stress"] / 1000.0  # kPa -> MPa
    df["roughness"] = df["rough"]

    cat = [c for c in ["printer", "material", "pattern"] if c in df]
    num = ["layer", "infill", "speed"] + ([c for c in ["width", "spec_t"] if c in df] if use_specimen_dims else [])
    features = cat + num
    group_cols = [c for c in ["printer", "material", "layer", "infill", "pattern", "speed"] if c in df]

    before = len(df)
    df = df.dropna(subset=features + ["tensile", "roughness"]).reset_index(drop=True)
    if verbose:
        print(f"Rows kept: {len(df)} of {before}")

    bad = df.roughness <= 0  # a measured average roughness of 0 µm is physically impossible
    if bad.any():
        if verbose:
            print(f"Rows with roughness <= 0: {int(bad.sum())}" + (" (removed)" if drop_zero_roughness else ""))
        if drop_zero_roughness:
            df = df.loc[~bad].reset_index(drop=True)

    df["group"] = df[group_cols].astype(str).agg("|".join, axis=1)
    groups = df["group"].factorize()[0]
    config = {"FEATURES": features, "CAT": cat, "NUM": num, "GROUP_COLS": group_cols}
    return df, groups, config
