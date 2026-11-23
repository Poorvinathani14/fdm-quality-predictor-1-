"""Suggest print settings that maximise predicted tensile strength under a roughness limit.

Example:
    python scripts/optimize.py --data data/raw/dataset.xlsx --material "pla+" --roughness-limit 4.65
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fdm_quality import load_bundle, optimise, prepare, read_table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="outputs/fdm_quality_model.joblib")
    ap.add_argument("--material", default=None, help="Default: the most frequent material")
    ap.add_argument("--roughness-limit", type=float, default=None, help="µm. Default: dataset median")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    bundle = load_bundle(args.model)
    cfg = bundle["config"]
    df, _, _ = prepare(read_table(args.data), use_specimen_dims="width" in cfg["NUM"], verbose=False)
    ok, cand, measured = optimise(bundle["models"], df, cfg, args.material, args.roughness_limit)

    cols = [c for c in ["printer", "layer", "infill", "pattern", "speed", "pred_tensile", "pred_rough"] if c in ok]
    print(f"Candidates tried: {len(cand)} | allowed: {len(ok)}\n")
    print("Top suggestions (model predictions):")
    print(ok[cols].head(args.top).round(2).to_string(index=False))
    mcols = [c for c in ["printer", "layer", "infill", "pattern", "speed", "tensile", "roughness"] if c in measured]
    print("\nBest MEASURED specimens (real data, for comparison):")
    print(measured[mcols].head(5).round(2).to_string(index=False))
    print("\nNote: these are model suggestions; confirm the best few with real prints and tests.")


if __name__ == "__main__":
    main()
