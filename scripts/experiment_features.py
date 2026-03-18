"""
Feature engineering experiments for fish length regression.

Tests new derived features and different model configs to improve
prediction range and reduce mean-regression.

Usage:
    python scripts/experiment_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    USE_XGBOOST = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    USE_XGBOOST = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "output" / "features.csv"

RANDOM_STATE = 42
N_FOLDS = 5

# ── Original 14 features ────────────────────────────────────────────────────
BASE_FEATURES = [
    "fish_box_width", "fish_box_height", "fish_box_area", "fish_aspect_ratio",
    "fish_box_x_center", "fish_box_y_center", "fish_confidence",
    "person_detected", "person_box_height", "fish_to_person_ratio",
    "species_index", "species_confidence", "image_aspect_ratio", "diagonal_fraction",
]


def add_engineered_features(df):
    """Add new derived features and return updated df + feature name list."""
    df = df.copy()

    # ── Interaction features ─────────────────────────────────────────────
    # Fish size relative to person in multiple ways
    df["fish_w_to_person_h"] = df["fish_box_width"] / df["person_box_height"].clip(lower=1)
    df["fish_h_to_person_h"] = df["fish_box_height"] / df["person_box_height"].clip(lower=1)
    df["fish_area_to_person_h_sq"] = df["fish_box_area"] / (df["person_box_height"].clip(lower=1) ** 2)

    # Fish pixel length (max of w,h) — the raw signal the baseline uses
    df["fish_pixel_length"] = df[["fish_box_width", "fish_box_height"]].max(axis=1)

    # Fish pixel length relative to person
    df["pixel_length_to_person"] = df["fish_pixel_length"] / df["person_box_height"].clip(lower=1)

    # ── Position features ────────────────────────────────────────────────
    # Fish center relative to person center (if person detected)
    # Y-position difference could indicate distance/angle
    df["fish_y_normalized"] = df["fish_box_y_center"] / 640.0
    df["fish_x_normalized"] = df["fish_box_x_center"] / 640.0

    # How much of the frame height does the fish occupy
    df["fish_height_fraction"] = df["fish_box_height"] / 640.0
    df["fish_width_fraction"] = df["fish_box_width"] / 640.0

    # ── Geometric features ───────────────────────────────────────────────
    # Perimeter proxy
    df["fish_perimeter"] = 2 * (df["fish_box_width"] + df["fish_box_height"])

    # Log transforms (can help with skewed distributions)
    df["log_fish_area"] = np.log1p(df["fish_box_area"])
    df["log_fish_pixel_length"] = np.log1p(df["fish_pixel_length"])
    df["log_person_height"] = np.log1p(df["person_box_height"])

    # Squared terms (help model fit non-linear relationships)
    df["fish_pixel_length_sq"] = df["fish_pixel_length"] ** 2
    df["fish_to_person_ratio_sq"] = df["fish_to_person_ratio"] ** 2

    new_features = [
        "fish_w_to_person_h", "fish_h_to_person_h", "fish_area_to_person_h_sq",
        "fish_pixel_length", "pixel_length_to_person",
        "fish_y_normalized", "fish_x_normalized",
        "fish_height_fraction", "fish_width_fraction",
        "fish_perimeter",
        "log_fish_area", "log_fish_pixel_length", "log_person_height",
        "fish_pixel_length_sq", "fish_to_person_ratio_sq",
    ]

    return df, BASE_FEATURES + new_features


def run_cv(X, y, baseline, label="", verbose=True):
    """Run 5-fold CV and return metrics."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        if USE_XGBOOST:
            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
            )
            model.fit(X[train_idx], y[train_idx],
                      eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        else:
            model = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=3, random_state=RANDOM_STATE,
            )
            model.fit(X[train_idx], y[train_idx])

        oof[val_idx] = model.predict(X[val_idx])

    mae = mean_absolute_error(y, oof)
    rmse = np.sqrt(mean_squared_error(y, oof))
    pred_range = oof.max() - oof.min()
    baseline_mae = mean_absolute_error(y, baseline)

    # Bias at extremes
    small_mask = y < np.percentile(y, 25)
    large_mask = y > np.percentile(y, 75)
    small_bias = (oof[small_mask] - y[small_mask]).mean() if small_mask.sum() > 0 else 0
    large_bias = (oof[large_mask] - y[large_mask]).mean() if large_mask.sum() > 0 else 0

    if verbose:
        imp_pct = (1 - mae / baseline_mae) * 100
        print(f"  {label:<45} MAE={mae:.2f}\"  RMSE={rmse:.2f}\"  range={pred_range:.1f}\"  "
              f"small_bias={small_bias:+.1f}\"  large_bias={large_bias:+.1f}\"  imp={imp_pct:.0f}%")

    return mae, rmse, oof, pred_range, small_bias, large_bias


def main():
    if not FEATURES_CSV.exists():
        sys.exit(f"Features CSV not found: {FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV)
    y = df["length_inches"].values
    baseline = df["baseline_prediction"].values

    print(f"Loaded {len(df)} samples")
    print(f"Actual length range: {y.min():.0f}-{y.max():.0f}\"  mean={y.mean():.1f}\"  std={y.std():.1f}\"")
    print(f"Baseline MAE: {mean_absolute_error(y, baseline):.2f}\"")
    print()

    # ── Experiment 1: Original 14 features ───────────────────────────────
    print("=" * 110)
    print("EXPERIMENT RESULTS")
    print("=" * 110)

    X_base = df[BASE_FEATURES].values
    run_cv(X_base, y, baseline, "1. Original 14 features")

    # ── Experiment 2: Original + engineered features ─────────────────────
    df_eng, all_feature_names = add_engineered_features(df)
    X_all = df_eng[all_feature_names].values
    run_cv(X_all, y, baseline, "2. Original + 15 engineered (29 total)")

    # ── Experiment 3: Just the best ratio-based features ─────────────────
    ratio_features = [
        "fish_pixel_length", "pixel_length_to_person",
        "fish_w_to_person_h", "fish_h_to_person_h",
        "fish_to_person_ratio", "fish_aspect_ratio",
        "person_box_height", "fish_confidence",
        "species_index", "diagonal_fraction",
    ]
    X_ratio = df_eng[ratio_features].values
    run_cv(X_ratio, y, baseline, "3. Curated ratio features (10)")

    # ── Experiment 4: Shallower trees (less overfitting) ─────────────────
    print()
    print("  --- Depth experiments (all 29 features) ---")
    for depth in [2, 3, 4, 5, 6]:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros(len(y))
        for _, (tr, va) in enumerate(kf.split(X_all)):
            if USE_XGBOOST:
                m = xgb.XGBRegressor(
                    n_estimators=300, max_depth=depth, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                    reg_alpha=0.5, reg_lambda=2.0, random_state=RANDOM_STATE,
                )
                m.fit(X_all[tr], y[tr], eval_set=[(X_all[va], y[va])], verbose=False)
            else:
                m = GradientBoostingRegressor(
                    n_estimators=300, max_depth=depth, learning_rate=0.03,
                    subsample=0.8, min_samples_leaf=3, random_state=RANDOM_STATE,
                )
                m.fit(X_all[tr], y[tr])
            oof[va] = m.predict(X_all[va])

        mae = mean_absolute_error(y, oof)
        pred_range = oof.max() - oof.min()
        small_mask = y < np.percentile(y, 25)
        large_mask = y > np.percentile(y, 75)
        sb = (oof[small_mask] - y[small_mask]).mean()
        lb = (oof[large_mask] - y[large_mask]).mean()
        print(f"  depth={depth}  n_est=300  lr=0.03               "
              f"MAE={mae:.2f}\"  range={pred_range:.1f}\"  "
              f"small_bias={sb:+.1f}\"  large_bias={lb:+.1f}\"")

    # ── Experiment 5: Feature importance with best config ────────────────
    print()
    print("  --- Feature importances (29 features, best config) ---")
    if USE_XGBOOST:
        final = xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            reg_alpha=0.5, reg_lambda=2.0, random_state=RANDOM_STATE,
        )
    else:
        final = GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=3, random_state=RANDOM_STATE,
        )
    final.fit(X_all, y)
    importances = sorted(zip(all_feature_names, final.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances[:15]:
        bar = "█" * int(imp * 200)
        print(f"    {feat:<35} {imp:.4f}  {bar}")

    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
