# FDM 3D-Printing Quality Predictor

Predict **tensile strength (MPa)** and **surface roughness (µm)** of FDM-printed parts from print settings (printer type, material, infill pattern, layer thickness, infill density, print speed), and search for settings that maximise strength under a roughness limit.

Converted from a Google Colab notebook into an installable-style Python package, command-line scripts and a cleaned notebook.

## Results (original notebook, held-out test set)

| Target | Model | Test R² | MAE | RMSE |
|---|---|---|---|---|
| Tensile strength | Random Forest | 0.854 | 2.010 | 2.803 |
| Surface roughness | Random Forest | 0.725 | 0.827 | 1.127 |

Linear regression reached R² 0.785 / 0.685, and a mean-only baseline about -0.07 / -0.01. Most influential inputs: material, layer thickness and infill density for strength; layer thickness, material and print speed for roughness.

Evaluation is **group-aware**: rows sharing one print-setting combination never appear in both train and test, so repeated specimens don't leak. The best model is chosen by grouped 5-fold CV R², not test R².

## Layout

```
src/fdm_quality/   data.py, modeling.py, optimize.py, predict.py
scripts/           train.py, predict.py, optimize.py
notebooks/         cleaned original notebook (outputs stripped)
data/              download instructions (data not committed)
```

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset (see [data/README.md](data/README.md)) into `data/raw/`.

## Usage

```bash
# 1. Train, evaluate, save plots and model to outputs/
python scripts/train.py --data data/raw/<file>.xlsx

# 2. Predict for one set of settings
python scripts/predict.py --material "pla+" --layer 0.2 --infill 80 --pattern gyroid --speed 50

# 3. Suggest settings (max strength under a roughness limit)
python scripts/optimize.py --data data/raw/<file>.xlsx --material "pla+" --roughness-limit 4.65
```

Add `--use-specimen-dims` to `train.py` to include specimen width and thickness (these are measured after printing, so they are off by default).

From Python:

```python
import sys; sys.path.insert(0, "src")
from fdm_quality import load_bundle, predict_quality
bundle = load_bundle("outputs/fdm_quality_model.joblib")
predict_quality(bundle, "pla+", 0.2, 80, "gyroid", 50)
```

Predictions warn when inputs fall outside the training range or use unseen categories. Optimisation only tries setting values observed for the chosen material, and its output should be confirmed with real prints.

## Data and license

Code: MIT (see `LICENSE`). Dataset: Aktepe & Ergün (2026), Mendeley Data, DOI 10.17632/zd6td6svd6.2, CC BY 4.0.
