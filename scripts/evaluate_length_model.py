"""
Evaluate the length regression model vs the heuristic baseline.

Generates comparison plots and a printable summary.

Usage:
    python scripts/evaluate_length_model.py

Expects:
    output/oof_predictions.csv   — from train_length_regressor.py
    output/training_results.json — from train_length_regressor.py

Outputs:
    output/eval_predicted_vs_actual.png
    output/eval_error_distribution.png
    output/eval_per_species_mae.png
    output/eval_feature_importances.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OOF_CSV = OUTPUT_DIR / "oof_predictions.csv"
RESULTS_JSON = OUTPUT_DIR / "training_results.json"


def plot_predicted_vs_actual(df, metrics):
    """Scatter plot: model vs baseline predictions against ground truth."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    y = df["length_inches"]
    model_pred = df["oof_prediction"]
    baseline_pred = df["baseline_prediction"]

    lo = min(y.min(), model_pred.min(), baseline_pred.min()) - 2
    hi = max(y.max(), model_pred.max(), baseline_pred.max()) + 2

    # Model
    ax1.scatter(y, model_pred, alpha=0.7, s=60, edgecolors="k", linewidth=0.5)
    ax1.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect")
    ax1.set_xlabel("Actual Length (inches)", fontsize=12)
    ax1.set_ylabel("Model Predicted (inches)", fontsize=12)
    ax1.set_title(f"XGBoost Model  (MAE={metrics['overall']['model_mae']:.2f}\")", fontsize=13, weight="bold")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Baseline
    ax2.scatter(y, baseline_pred, alpha=0.7, s=60, color="orange", edgecolors="k", linewidth=0.5)
    ax2.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect")
    ax2.set_xlabel("Actual Length (inches)", fontsize=12)
    ax2.set_ylabel("Baseline Predicted (inches)", fontsize=12)
    ax2.set_title(f"Heuristic Baseline  (MAE={metrics['overall']['baseline_mae']:.2f}\")", fontsize=13, weight="bold")
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_aspect("equal")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "eval_predicted_vs_actual.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_error_distribution(df, metrics):
    """Side-by-side histograms of model vs baseline errors."""
    model_errors = df["oof_prediction"] - df["length_inches"]
    baseline_errors = df["baseline_prediction"] - df["length_inches"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(
        min(model_errors.min(), baseline_errors.min()) - 1,
        max(model_errors.max(), baseline_errors.max()) + 1,
        25,
    )

    ax1.hist(model_errors, bins=bins, alpha=0.7, edgecolor="black", color="steelblue")
    ax1.axvline(0, color="red", linestyle="--", linewidth=2)
    ax1.set_xlabel("Error (inches)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(f"Model Errors  (MAE={metrics['overall']['model_mae']:.2f}\")", fontsize=13, weight="bold")
    ax1.grid(alpha=0.3)

    ax2.hist(baseline_errors, bins=bins, alpha=0.7, edgecolor="black", color="orange")
    ax2.axvline(0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Error (inches)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(f"Baseline Errors  (MAE={metrics['overall']['baseline_mae']:.2f}\")", fontsize=13, weight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "eval_error_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_per_species_mae(metrics):
    """Bar chart of MAE per species: model vs baseline."""
    per_species = metrics.get("per_species", {})
    if not per_species:
        print("  Skipping per-species plot (no species data)")
        return

    species = list(per_species.keys())
    model_maes = [per_species[s]["model_mae"] for s in species]
    baseline_maes = [per_species[s]["baseline_mae"] for s in species]
    counts = [per_species[s]["n_samples"] for s in species]

    # Sort by baseline MAE descending
    order = np.argsort(baseline_maes)[::-1]
    species = [species[i] for i in order]
    model_maes = [model_maes[i] for i in order]
    baseline_maes = [baseline_maes[i] for i in order]
    counts = [counts[i] for i in order]

    x = np.arange(len(species))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, baseline_maes, width, label="Baseline", color="orange", alpha=0.8)
    bars2 = ax.bar(x + width / 2, model_maes, width, label="Model", color="steelblue", alpha=0.8)

    # Add count labels
    for i, count in enumerate(counts):
        ax.text(i, max(baseline_maes[i], model_maes[i]) + 0.3, f"n={count}", ha="center", fontsize=9)

    ax.set_xlabel("Species", fontsize=12)
    ax.set_ylabel("MAE (inches)", fontsize=12)
    ax.set_title("MAE by Species: Model vs Baseline", fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = OUTPUT_DIR / "eval_per_species_mae.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importances(metrics):
    """Horizontal bar chart of feature importances."""
    importances = metrics.get("feature_importances", {})
    if not importances:
        return

    features = list(importances.keys())
    values = list(importances.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color="steelblue", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Feature Importances (XGBoost)", fontsize=14, weight="bold")
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    path = OUTPUT_DIR / "eval_feature_importances.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    if not OOF_CSV.exists():
        sys.exit(f"OOF predictions not found: {OOF_CSV}\nRun train_length_regressor.py first.")
    if not RESULTS_JSON.exists():
        sys.exit(f"Training results not found: {RESULTS_JSON}\nRun train_length_regressor.py first.")

    df = pd.read_csv(OOF_CSV)
    with open(RESULTS_JSON) as f:
        metrics = json.load(f)

    o = metrics["overall"]
    print(f"{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Samples:      {o['n_samples']}")
    print(f"  Model MAE:    {o['model_mae']:.2f}\"")
    print(f"  Baseline MAE: {o['baseline_mae']:.2f}\"")
    print(f"  Improvement:  {o['improvement_pct']:.1f}%")
    print(f"  Model RMSE:   {o['model_rmse']:.2f}\"")
    print(f"  Model R²:     {o['model_r2']:.3f}")

    wp = metrics.get("with_person", {})
    wop = metrics.get("without_person", {})
    if wp.get("n_samples", 0) > 0:
        print(f"\n  With person ({wp['n_samples']}):    model={wp['model_mae']:.2f}\"  baseline={wp['baseline_mae']:.2f}\"")
    if wop.get("n_samples", 0) > 0:
        print(f"  Without person ({wop['n_samples']}): model={wop['model_mae']:.2f}\"  baseline={wop['baseline_mae']:.2f}\"")

    print(f"\nGenerating plots...")
    plot_predicted_vs_actual(df, metrics)
    plot_error_distribution(df, metrics)
    plot_per_species_mae(metrics)
    plot_feature_importances(metrics)

    # Promotion gate
    gate = "PASS" if o["improvement_pct"] > 10 else "MARGINAL" if o["improvement_pct"] > 0 else "FAIL"
    print(f"\n{'='*70}")
    print(f"  PROMOTION GATE: {gate}")
    if gate == "PASS":
        print(f"  Model is ready for Phase 2 deployment.")
    elif gate == "MARGINAL":
        print(f"  Model shows some improvement. Consider more data before deploying.")
    else:
        print(f"  Model does not beat baseline. Investigate features or gather more data.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
