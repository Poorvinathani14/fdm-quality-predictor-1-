"""Model building, evaluation and permutation importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

PARAM_GRID = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 8, 14],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 1.0],
}


def _prep(cat, num, scale: bool):
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat),
         ("num", StandardScaler() if scale else "passthrough", num)],
        sparse_threshold=0,
    )


def build(kind: str, cat, num, **rf_params):
    """kind='lin' -> Linear Regression pipeline; anything else -> Random Forest pipeline."""
    if kind == "lin":
        return Pipeline([("prep", _prep(cat, num, True)), ("model", LinearRegression())])
    params = {"n_estimators": 300, **rf_params}
    return Pipeline([("prep", _prep(cat, num, False)),
                     ("model", RandomForestRegressor(random_state=SEED, **params))])


def group_split(df, groups, test_size=0.2):
    """Group-aware split: every unique combination of print settings stays on one side."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    return next(gss.split(df, groups=groups))


def train_and_evaluate(df, groups, config, targets, tr_idx, te_idx, n_jobs=-1):
    """Train mean/Linear/RF/tuned-RF per target. Returns (results_df, fitted, best_params, best_name)."""
    features, cat, num = config["FEATURES"], config["CAT"], config["NUM"]
    Xtr, Xte = df.loc[tr_idx, features], df.loc[te_idx, features]
    gtr = groups[tr_idx]
    cv = GroupKFold(n_splits=5)

    rows, fitted, best_params, best_name = [], {}, {}, {}
    for tname, tcol in targets.items():
        ytr, yte = df.loc[tr_idx, tcol].values, df.loc[te_idx, tcol].values
        models = {
            "A. Mean (reference)": DummyRegressor(strategy="mean"),
            "B. Linear Regression": build("lin", cat, num),
            "C. Random Forest (default)": build("rf", cat, num),
        }
        cvs = {}
        for k, m in models.items():
            s = cross_val_score(m, Xtr, ytr, cv=cv, groups=gtr, scoring="r2")
            cvs[k] = (s.mean(), s.std())
            m.fit(Xtr, ytr)

        # Tuning uses the TRAINING part only
        gs = GridSearchCV(build("rf", cat, num), PARAM_GRID, cv=cv, scoring="r2", n_jobs=n_jobs)
        gs.fit(Xtr, ytr, groups=gtr)
        models["D. Random Forest (tuned)"] = gs.best_estimator_
        cvs["D. Random Forest (tuned)"] = (gs.best_score_, gs.cv_results_["std_test_score"][gs.best_index_])
        best_params[tname] = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}

        for k, m in models.items():
            p = m.predict(Xte)
            rows.append({
                "Target": tname, "Model": k,
                "CV R2 (mean)": cvs[k][0], "CV R2 (std)": cvs[k][1],
                "Test MAE": mean_absolute_error(yte, p),
                "Test RMSE": np.sqrt(mean_squared_error(yte, p)),
                "Test R2": r2_score(yte, p),
            })
        # chosen on cross-validation, NOT on the test set
        best_name[tname] = max([k for k in models if k[0] in "BCD"], key=lambda k: cvs[k][0])
        fitted[tname] = models
    return pd.DataFrame(rows).round(3), fitted, best_params, best_name


def importances(fitted, best_name, df, te_idx, config, targets, n_repeats=20):
    """Permutation importance on the held-out test set, per target."""
    features = config["FEATURES"]
    Xte = df.loc[te_idx, features]
    out = {}
    for tname, tcol in targets.items():
        m = fitted[tname][best_name[tname]]
        pi = permutation_importance(m, Xte, df.loc[te_idx, tcol].values,
                                    n_repeats=n_repeats, random_state=SEED, scoring="r2")
        out[tname] = pd.DataFrame({"mean": pi.importances_mean, "std": pi.importances_std},
                                  index=features).sort_values("mean", ascending=False)
    return out


def refit_all(fitted, best_name, df, config, targets):
    """Refit the chosen model for each target on ALL rows (deployment step, after evaluation)."""
    from sklearn.base import clone
    return {t: clone(fitted[t][best_name[t]]).fit(df[config["FEATURES"]], df[c]) for t, c in targets.items()}
