"""Save / load the trained models and predict quality for new settings."""
from __future__ import annotations

import joblib
import pandas as pd

from .data import LABEL


def save_bundle(path, final_models, config, targets, df):
    """Persist models plus the training ranges/categories needed for sanity warnings."""
    ranges = {c: (float(df[c].min()), float(df[c].max())) for c in ["layer", "infill", "speed"]}
    categories = {c: sorted(df[c].unique()) for c in config["CAT"]}
    defaults = {"printer": df["printer"].mode()[0] if "printer" in df else None,
                "width": float(df["width"].median()) if "width" in df else None,
                "spec_t": float(df["spec_t"].median()) if "spec_t" in df else None}
    joblib.dump({"models": final_models, "config": config, "targets": targets,
                 "ranges": ranges, "categories": categories, "defaults": defaults}, path)


def load_bundle(path):
    return joblib.load(path)


def predict_quality(bundle, material, layer, infill, pattern, speed, printer=None, width=None, spec_t=None):
    """Predict tensile strength (MPa) and roughness (µm); warn when inputs fall outside the training data."""
    cfg, defaults = bundle["config"], bundle["defaults"]
    norm = lambda v: " ".join(str(v).strip().lower().split())
    row = {"material": norm(material), "layer": layer, "infill": infill,
           "pattern": norm(pattern), "speed": speed}
    if "printer" in cfg["CAT"]:
        row["printer"] = norm(printer) if printer is not None else defaults["printer"]
    if "width" in cfg["NUM"]:
        row["width"] = width if width is not None else defaults["width"]
    if "spec_t" in cfg["NUM"]:
        row["spec_t"] = spec_t if spec_t is not None else defaults["spec_t"]
    X = pd.DataFrame([row])[cfg["FEATURES"]]

    out = {t: round(float(m.predict(X)[0]), 2) for t, m in bundle["models"].items()}
    warnings = []
    for c, (lo, hi) in bundle["ranges"].items():
        if not (lo <= row[c] <= hi):
            warnings.append(f"{LABEL[c]} {row[c]} is outside the training range {lo} to {hi}")
    for c, seen in bundle["categories"].items():
        if row[c] not in seen:
            warnings.append(f"{LABEL[c]} '{row[c]}' was never seen in the training data")
    out["warnings"] = warnings
    return out
