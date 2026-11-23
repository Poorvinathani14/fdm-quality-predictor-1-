"""Predict tensile strength and roughness for one set of print settings using a saved model.

Example:
    python scripts/predict.py --model outputs/fdm_quality_model.joblib \
        --material "pla+" --layer 0.2 --infill 80 --pattern gyroid --speed 50 --printer closed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fdm_quality import load_bundle, predict_quality


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="outputs/fdm_quality_model.joblib")
    ap.add_argument("--material", required=True)
    ap.add_argument("--layer", type=float, required=True, help="Layer thickness (mm)")
    ap.add_argument("--infill", type=float, required=True, help="Infill density (%%)")
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--speed", type=float, required=True, help="Print speed (mm/s)")
    ap.add_argument("--printer", default=None, help="open / closed")
    args = ap.parse_args()

    bundle = load_bundle(args.model)
    out = predict_quality(bundle, args.material, args.layer, args.infill, args.pattern,
                          args.speed, printer=args.printer)
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
