"""Search for print settings that maximise predicted strength under a roughness limit."""
from __future__ import annotations

import itertools

import pandas as pd

from .modeling import SEED

T_TENSILE, T_ROUGH = "Tensile strength (MPa)", "Surface roughness (µm)"


def optimise(final_models, df, config, material=None, roughness_limit=None, max_candidates=100_000):
    """Grid-search OBSERVED setting values for one material.

    Returns (allowed_candidates_sorted, all_candidates, best_measured_specimens).
    Only values actually tested for the material are tried, so the model never extrapolates.
    """
    features, num = config["FEATURES"], config["NUM"]
    material = material or df.material.value_counts().index[0]
    roughness_limit = float(df.roughness.median()) if roughness_limit is None else roughness_limit

    obs = df[df.material == material]
    if obs.empty:
        raise ValueError(f"Material '{material}' not found. Available: {sorted(df.material.unique())}")
    varied = ["layer", "infill", "speed", "pattern"] + (["printer"] if "printer" in df else [])
    levels = {c: sorted(obs[c].unique()) for c in varied}
    cand = pd.DataFrame(list(itertools.product(*levels.values())), columns=list(levels))
    cand["material"] = material
    for c in ["width", "spec_t"]:
        if c in num:
            cand[c] = obs[c].median()
    if len(cand) > max_candidates:
        cand = cand.sample(max_candidates, random_state=SEED)
    cand = cand[features].reset_index(drop=True)
    cand["pred_tensile"] = final_models[T_TENSILE].predict(cand[features])
    cand["pred_rough"] = final_models[T_ROUGH].predict(cand[features])

    ok = cand[cand.pred_rough <= roughness_limit].sort_values("pred_tensile", ascending=False)
    measured = obs[obs.roughness <= roughness_limit].sort_values("tensile", ascending=False)
    return ok.reset_index(drop=True), cand, measured
