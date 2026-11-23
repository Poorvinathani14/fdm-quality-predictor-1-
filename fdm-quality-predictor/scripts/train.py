"""Train the models, print the metrics table, save plots and the model bundle.

Usage:
    python scripts/train.py --data data/raw/dataset.xlsx --out outputs
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fdm_quality import (TARGETS, group_split, importances, prepare, read_table, refit_all,
                         save_bundle, train_and_evaluate)
from fdm_quality.data import LABEL


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Path to the Mendeley .xlsx or .csv file")
    ap.add_argument("--out", default="outputs", help="Output folder")
    ap.add_argument("--use-specimen-dims", action="store_true",
                    help="Also use specimen width/thickness as inputs (measured AFTER printing)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = read_table(args.data)
    df, groups, cfg = prepare(raw, use_specimen_dims=args.use_specimen_dims)
    tr_idx, te_idx = group_split(df, groups)
    print(f"Train rows: {len(tr_idx)} | Test rows: {len(te_idx)}")

    res, fitted, best_params, best_name = train_and_evaluate(df, groups, cfg, TARGETS, tr_idx, te_idx)
    res.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    for t in TARGETS:
        print("\n" + t)
        print(res[res.Target == t].drop(columns="Target").to_string(index=False))
        print("tuned parameters:", best_params[t])

    imp = importances(fitted, best_name, df, te_idx, cfg, TARGETS)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, (t, d) in zip(ax, imp.items()):
        d = d.sort_values("mean")
        a.barh([LABEL[c] for c in d.index], d["mean"], xerr=d["std"])
        a.set_xlabel("Drop in test R2 when the input is shuffled")
        a.set_title(f"{t}\n({best_name[t]})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "permutation_importance.png"), dpi=150)

    final = refit_all(fitted, best_name, df, cfg, TARGETS)
    model_path = os.path.join(args.out, "fdm_quality_model.joblib")
    save_bundle(model_path, final, cfg, TARGETS, df)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()
