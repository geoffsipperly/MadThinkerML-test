# MadThinkerML — Fish Length Estimation

Replaces the fixed `pixelsPerInch` heuristic in the EpicWaters app with a learned regression model that estimates fish length from YOLO detection features.

## Current Results (Phase 1)

| Metric | Baseline (heuristic) | Model (GradientBoosting) |
|--------|---------------------|--------------------------|
| MAE | 8.16" | **2.12"** |
| RMSE | — | 2.78" |
| Improvement | — | **74.0%** |
| R² | — | 0.071 |

Per-species:
- `steelhead_traveler` (n=71): **1.94"** MAE
- `steelhead_holding` (n=13): **3.14"** MAE

## How It Works

1. **YOLO** detects fish and person bounding boxes (640x640 space)
2. **ViT** classifies species
3. **15 features** are extracted: fish geometry, person-as-reference ratios, species index, confidence scores
4. **Gradient-boosted regressor** predicts length in inches

Top features driving predictions:
- `fish_area_to_person_h_sq` (17.9%) — fish area normalized by person height²
- `fish_box_height` (12.0%)
- `fish_h_to_person_h` (9.5%) — fish height relative to person height
- `fish_w_to_person_h` (9.4%)
- `fish_box_area` (8.3%)

## Project Structure

```
MadThinkerML/
├── scripts/
│   ├── extract_features.py        # Run YOLO + ViT → 15-feature CSV
│   ├── train_length_regressor.py  # Train model, 5-fold CV, export
│   ├── evaluate_length_model.py   # Comparison plots, promotion gate
│   ├── experiment_features.py     # Feature engineering experiments
│   └── extract_unlabeled.py       # Feature extraction for unlabeled photos
├── models/                        # Trained model artifacts (.pkl, .mlpackage)
├── data/
│   ├── ground_truth/
│   │   ├── images/                # Labeled fish photos
│   │   └── labels.csv             # filename, length_inches, species
│   └── unlabeled/
│       └── images/                # Unlabeled fish photos for distribution analysis
├── output/                        # Features CSV, predictions, plots, metrics
└── requirements.txt
```

## Usage

```bash
# Set up
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python scripts/extract_features.py          # Extract features from labeled photos
python scripts/train_length_regressor.py    # Train and evaluate model
python scripts/evaluate_length_model.py     # Generate comparison plots

# Optional
python scripts/experiment_features.py       # Run feature engineering experiments
python scripts/extract_unlabeled.py         # Extract features from unlabeled photos
```

## Data Requirements

Place labeled photos in `data/ground_truth/`:
- `images/` — fish photos (JPG/PNG)
- `labels.csv` — CSV with columns: `Image Filename`, `Final Length`, `Final Species`
  (or simple format: `filename`, `length_inches`, `species`)

Models from EpicWatersML are referenced at `~/dev/EpicWatersML/`:
- YOLO: `runs/fish_det/steelhead_fish_v3/weights/best.pt`
- ViT: `vit_fish_species_tiny_best.pt`

## Known Limitations

- **Mean-regression at extremes**: Model predicts in ~27-35" range vs actual 25-40". Fish <27" are overestimated, >35" underestimated. Root cause: limited training data at the tails.
- **All photos have a person**: No training data yet for fish-only photos (no person as scale reference).
- **Species imbalance**: 85% steelhead_traveler, 15% steelhead_holding. More species diversity needed.

## Roadmap

- **Phase 2**: Deploy model to iOS app, capture richer feature data in Supabase
- **Phase 3**: Server-side endpoint for Android, automated retraining pipeline
- **Phase 4**: YOLO keypoint detection (nose-to-tail) for true length measurement
