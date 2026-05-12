"""
train_minutes.py
────────────────
Train a dedicated XGBoost Poisson regressor to project tonight's minutes.

The predicted minutes are fed back into the main pts/reb/ast models as
`l5_min`, replacing the historical L5 average. This improves projections in
blowout-risk games (large spread → fewer star minutes) and high-pace games
(higher total → more possessions → potentially more minutes).

Features:
  Historical:  l5_min, std_min, gp_prior
  Game-level:  is_home, rest_days
  Market-info: (game_total, spread_abs) — added when historical odds available

Outputs:
    xgb_min_model.json
    model_meta_min.json
"""

import json
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Config ────────────────────────────────────────────────────────────────────
TRAINING_FILE = "training_data.parquet"
TRAIN_SEASONS = ["2022-23", "2023-24"]
VAL_SEASONS   = ["2024-25"]

FEATURE_COLS = [
    "l5_min", "std_min",
    "l10_min_std",
    "ewma_min",           # EWMA recency — captures recent trend in minutes
    "gp_prior", "is_home", "rest_days",
    "opp_pace_roll10",    # fast-paced game → higher minutes ceiling
    "opp_def_roll10",     # elite defense → closer game → more star minutes
]

TARGET = "MIN"   # actual minutes played that game (the column name in training_data.parquet)

XGB_PARAMS = dict(
    objective          = "count:poisson",
    eval_metric        = "rmse",
    max_depth          = 4,
    eta                = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_weight   = 10,
    n_estimators       = 600,
    early_stopping_rounds = 40,
    tree_method        = "hist",
    random_state       = 42,
)


def _impute(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    df = df.copy()
    for col, med in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)
    return df


def main():
    df = pd.read_parquet(TRAINING_FILE)
    print(f"Loaded {len(df):,} rows | seasons: {sorted(df['season'].unique())}")

    if TARGET not in df.columns:
        print(f"ERROR: '{TARGET}' column not found. Add it to training_data.parquet.")
        print(f"Available columns: {list(df.columns)}")
        return

    # Filter rows where target minutes are valid (> 0 = actually played)
    df = df[df[TARGET] > 0].copy()

    avail_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠ Missing features (will be zero-imputed): {missing}")

    train_mask = df["season"].isin(TRAIN_SEASONS)
    val_mask   = df["season"].isin(VAL_SEASONS)
    train_raw  = df[train_mask]
    val_raw    = df[val_mask]

    medians = {c: float(train_raw[c].median()) for c in avail_features if c in train_raw.columns}
    # Zero-impute missing feature columns
    for col in missing:
        train_raw = train_raw.copy(); train_raw[col] = 0.0
        val_raw   = val_raw.copy();   val_raw[col]   = 0.0
        medians[col] = 0.0

    train = _impute(train_raw, medians)
    val   = _impute(val_raw,   medians)

    print(f"Train: {len(train):,} rows | Val: {len(val):,} rows")

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values.clip(min=0)
    X_val   = val[FEATURE_COLS].values
    y_val   = val[TARGET].values.clip(min=0)

    n_est = XGB_PARAMS.pop("n_estimators")
    model = xgb.XGBRegressor(n_estimators=n_est, **XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    preds    = model.predict(X_val).clip(min=0)
    rmse     = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae      = float(mean_absolute_error(y_val, preds))
    within_3 = float(np.mean(np.abs(preds - y_val) <= 3.0))
    within_5 = float(np.mean(np.abs(preds - y_val) <= 5.0))

    print(f"\n[Minutes] Val RMSE: {rmse:.3f}  MAE: {mae:.3f}  ±3min: {within_3:.1%}  ±5min: {within_5:.1%}")

    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("Top features:")
    for feat, imp in fi.head(6).items():
        print(f"  {feat:<20} {imp:.4f}")

    model.save_model("xgb_min_model.json")
    print("Saved → xgb_min_model.json")

    meta = {
        "feature_cols":  FEATURE_COLS,
        "medians":       medians,
        "train_seasons": TRAIN_SEASONS,
        "val_seasons":   VAL_SEASONS,
        "metrics":       {"rmse": rmse, "mae": mae, "within_3": within_3, "within_5": within_5},
        "best_iteration": int(model.best_iteration),
    }
    with open("model_meta_min.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved → model_meta_min.json")
    print("Done.")


if __name__ == "__main__":
    main()
