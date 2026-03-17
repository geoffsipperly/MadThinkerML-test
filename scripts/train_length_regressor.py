"""
Train a gradient-boosted regressor for fish length estimation.

Uses features extracted by extract_features.py. Performs 5-fold cross-validation,
reports MAE/RMSE/R² vs the heuristic baseline, and exports the model.

Tries XGBoost first; falls back to sklearn GradientBoostingRegressor if XGBoost
is unavailable (e.g. missing libomp on macOS).

Usage:
    python scripts/train_length_regressor.py

Expects:
    output/features.csv — from extract_features.py

Outputs:
    models/length_regressor.pkl        — trained model (pickle)
    models/length_regressor.mlpackage  — CoreML export (if coremltools available)
    output/training_results.json       — metrics summary
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    USE_XGBOOST = True
    print("Using XGBoost backend")
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    USE_XGBOOST = False
    print("XGBoost unavailable, using sklearn GradientBoostingRegressor")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "output" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# The 14 features used by the model
FEATURE_COLS = [
    "fish_box_width",
    "fish_box_height",
    "fish_box_area",
    "fish_aspect_ratio",
    "fish_box_x_center",
    "fish_box_y_center",
    "fish_confidence",
    "person_detected",
    "person_box_height",
    "fish_to_person_ratio",
    "species_index",
    "species_confidence",
    "image_aspect_ratio",
    "diagonal_fraction",
]

TARGET_COL = "length_inches"
BASELINE_COL = "baseline_prediction"

N_FOLDS = 5
RANDOM_STATE = 42


def train_and_evaluate(df):
    """5-fold cross-validated training. Returns fold metrics and final model."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    baseline = df[BASELINE_COL].values
    person_mask = df["person_detected"].values == 1.0

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = []
    oof_preds = np.zeros(len(y))
    oof_baseline = baseline.copy()

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if USE_XGBOOST:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        baseline_mae = mean_absolute_error(y_val, baseline[val_idx])

        fold_metrics.append({
            "fold": fold_idx + 1,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "baseline_mae": baseline_mae,
            "n_val": len(val_idx),
        })

        print(f"  Fold {fold_idx + 1}: MAE={mae:.2f}\"  RMSE={rmse:.2f}\"  R²={r2:.3f}  (baseline MAE={baseline_mae:.2f}\")")

    # Overall OOF metrics
    overall_mae = mean_absolute_error(y, oof_preds)
    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    overall_r2 = r2_score(y, oof_preds)
    overall_baseline_mae = mean_absolute_error(y, oof_baseline)

    # Split by person presence
    with_person = person_mask
    without_person = ~person_mask

    metrics = {
        "overall": {
            "model_mae": overall_mae,
            "model_rmse": overall_rmse,
            "model_r2": overall_r2,
            "baseline_mae": overall_baseline_mae,
            "improvement_pct": (1 - overall_mae / overall_baseline_mae) * 100 if overall_baseline_mae > 0 else 0,
            "n_samples": len(y),
        },
        "with_person": {
            "model_mae": mean_absolute_error(y[with_person], oof_preds[with_person]) if with_person.sum() > 0 else None,
            "baseline_mae": mean_absolute_error(y[with_person], baseline[with_person]) if with_person.sum() > 0 else None,
            "n_samples": int(with_person.sum()),
        },
        "without_person": {
            "model_mae": mean_absolute_error(y[without_person], oof_preds[without_person]) if without_person.sum() > 0 else None,
            "baseline_mae": mean_absolute_error(y[without_person], baseline[without_person]) if without_person.sum() > 0 else None,
            "n_samples": int(without_person.sum()),
        },
        "folds": fold_metrics,
    }

    # Per-species breakdown
    if "species_name" in df.columns:
        species_metrics = {}
        for species in df["species_name"].unique():
            mask = df["species_name"].values == species
            if mask.sum() < 2:
                continue
            species_metrics[species] = {
                "model_mae": float(mean_absolute_error(y[mask], oof_preds[mask])),
                "baseline_mae": float(mean_absolute_error(y[mask], baseline[mask])),
                "n_samples": int(mask.sum()),
            }
        metrics["per_species"] = species_metrics

    # Train final model on all data
    print("\nTraining final model on all data...")
    if USE_XGBOOST:
        final_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
        )
        final_model.fit(X, y, verbose=False)
    else:
        final_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3, random_state=RANDOM_STATE,
        )
        final_model.fit(X, y)

    # Feature importances
    importances = dict(zip(FEATURE_COLS, final_model.feature_importances_.tolist()))
    metrics["feature_importances"] = dict(sorted(importances.items(), key=lambda x: -x[1]))

    return final_model, metrics, oof_preds


def export_model(model, metrics):
    """Save pickle and attempt CoreML export."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Pickle
    pkl_path = MODELS_DIR / "length_regressor.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, f)
    print(f"  Pickle saved: {pkl_path}")

    # CoreML export
    try:
        import coremltools as ct

        if USE_XGBOOST:
            coreml_model = ct.converters.xgboost.convert(
                model, feature_names=FEATURE_COLS, target="length_inches",
            )
        else:
            coreml_model = ct.converters.sklearn.convert(
                model, input_features=FEATURE_COLS, output_feature_names="length_inches",
            )
        coreml_model.author = "MadThinkerML"
        coreml_model.short_description = "Fish length estimator from YOLO/ViT features"

        mlpackage_path = MODELS_DIR / "LengthRegressor.mlpackage"
        coreml_model.save(str(mlpackage_path))
        print(f"  CoreML saved: {mlpackage_path}")
    except ImportError:
        print("  CoreML export skipped (coremltools not installed)")
    except Exception as e:
        print(f"  CoreML export failed: {e}")


def main():
    if not FEATURES_CSV.exists():
        sys.exit(f"Features CSV not found: {FEATURES_CSV}\nRun extract_features.py first.")

    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} samples from {FEATURES_CSV}\n")

    if len(df) < 10:
        sys.exit(f"Too few samples ({len(df)}). Need at least 10 for cross-validation.")

    # Train
    print(f"Training with {N_FOLDS}-fold cross-validation...")
    model, metrics, oof_preds = train_and_evaluate(df)

    # Print summary
    o = metrics["overall"]
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Model MAE:    {o['model_mae']:.2f}\"")
    print(f"  Baseline MAE: {o['baseline_mae']:.2f}\"")
    print(f"  Improvement:  {o['improvement_pct']:.1f}%")
    print(f"  Model RMSE:   {o['model_rmse']:.2f}\"")
    print(f"  Model R²:     {o['model_r2']:.3f}")

    wp = metrics["with_person"]
    wop = metrics["without_person"]
    if wp["n_samples"] > 0:
        print(f"\n  With person ({wp['n_samples']} imgs):    model={wp['model_mae']:.2f}\"  baseline={wp['baseline_mae']:.2f}\"")
    if wop["n_samples"] > 0:
        print(f"  Without person ({wop['n_samples']} imgs): model={wop['model_mae']:.2f}\"  baseline={wop['baseline_mae']:.2f}\"")

    if "per_species" in metrics:
        print(f"\n  Per-species MAE:")
        for species, sm in metrics["per_species"].items():
            print(f"    {species:<30} model={sm['model_mae']:.2f}\"  baseline={sm['baseline_mae']:.2f}\"  (n={sm['n_samples']})")

    print(f"\n  Top features:")
    for feat, imp in list(metrics["feature_importances"].items())[:5]:
        print(f"    {feat:<30} {imp:.4f}")

    # Export model
    print(f"\nExporting model...")
    export_model(model, metrics)

    # Save OOF predictions for evaluate script
    df["oof_prediction"] = oof_preds
    oof_path = OUTPUT_DIR / "oof_predictions.csv"
    df.to_csv(oof_path, index=False)
    print(f"  OOF predictions saved: {oof_path}")

    # Save metrics JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics saved: {results_path}")

    print(f"\n{'='*70}")
    gate = "PASS" if o["improvement_pct"] > 10 else "MARGINAL" if o["improvement_pct"] > 0 else "FAIL"
    print(f"  GATE: {gate} ({o['improvement_pct']:.1f}% improvement over baseline)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
