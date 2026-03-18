# MadThinkerML — Fish Length Estimation

Replaces the fixed `pixelsPerInch` heuristic in the EpicWaters app with a learned regression model that estimates fish length from multi-source detection features.

## Current Results (Phase 1)

| Metric | Baseline (heuristic) | Model (GradientBoosting) |
|--------|---------------------|--------------------------|
| MAE | 8.16" | **2.20"** |
| RMSE | — | 2.78" |
| Improvement | — | **73.1%** |

Per-species:
- `steelhead_traveler` (n=71): **2.00"** MAE
- `steelhead_holding` (n=13): **3.26"** MAE

Per-context:
- Photos with hand detected (n=25): **2.05"** MAE
- Photos without hand (n=59): **2.26"** MAE

### Model Evolution

| Version | Features | MAE | Key Change |
|---------|----------|-----|------------|
| v1 | 14 base features | 2.33" | Initial YOLO + ViT features |
| v2 | 15 curated + engineered | 2.12" | Feature selection, person-ratio features |
| v3 | + person width/aspect ratio | 2.22" | Pose-invariant person reference |
| v4 | + MediaPipe finger detection | 2.18" | Finger as scale reference (unfiltered) |
| **v5** | **+ finger sanity filter** | **2.20"** | Reject bad hand detections, cleaner signal |

## How It Works

Three detection models feed features into a gradient-boosted regressor:

1. **YOLO** detects fish and person bounding boxes (640x640 space)
2. **ViT** classifies species (9 classes)
3. **MediaPipe Hands** detects finger landmarks for per-photo scale reference

From these detections, **26 features** are computed:

| Category | Features | Why |
|----------|----------|-----|
| Fish geometry | box width/height/area, aspect ratio, position, confidence | Primary size signal |
| Person reference | box height/width, aspect ratio, fish-to-person ratios | Scale reference (pose-dependent) |
| Finger reference | finger width/length in pixels, PPI from finger, fish-to-finger ratios | Scale reference at fish depth (pose-invariant) |
| Species | index + confidence from ViT | Body proportion context |
| Image geometry | diagonal fraction | Frame coverage |
| Engineered | fish_area_to_person_h_sq, fish_w_to_person_w, etc. | Scale-invariant combinations |

Top features by importance:
- `fish_area_to_person_h_sq` (17.2%) — fish area normalized by person height²
- `fish_box_height` (9.4%)
- `fish_h_to_person_h` (8.6%) — fish height relative to person height
- `fish_w_to_person_h` (6.8%)
- `fish_w_to_person_w` (6.7%) — fish width relative to person shoulder width

### Why Three Reference Signals?

The person bounding box gives a scale reference, but its height changes with pose (standing vs bent over). Person **width** (shoulder span) is more stable across poses. Finger measurements go further — they're at the same depth as the fish, so the fish-to-finger pixel ratio directly encodes real-world size regardless of camera distance. The model learns when to trust each signal.

### Finger Detection Sanity Filter

MediaPipe sometimes places hand landmarks on fish bodies, gloves, or other objects. A sanity filter rejects bad detections by checking:
- Finger length-to-width ratio is anatomically plausible (1.5-10x)
- Implied pixels-per-inch is reasonable for the image resolution (10-500)
- Implied maximum fish length from PPI is within 5-80"

This reduced hand detections from 55/84 (65%) to 25/84 (30%) but improved the finger-only MAE from 14.73" to 5.00".

## Project Structure

```
MadThinkerML/
├── scripts/
│   ├── extract_features.py        # Run YOLO + ViT + MediaPipe → feature CSV
│   ├── train_length_regressor.py  # Train model, 5-fold CV, export
│   ├── evaluate_length_model.py   # Comparison plots, promotion gate
│   ├── experiment_features.py     # Feature engineering experiments
│   └── extract_unlabeled.py       # Feature extraction for unlabeled photos
├── models/
│   ├── hand_landmarker.task       # MediaPipe hand landmark model
│   ├── length_regressor.pkl       # Trained regressor (gitignored)
│   └── LengthRegressor.mlpackage  # CoreML export (gitignored)
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

MediaPipe hand landmarker is included at `models/hand_landmarker.task`.

## Known Limitations

- **Mean-regression at extremes**: Model predicts in ~27-36" range vs actual 25-40". Fish <27" are overestimated, >35" underestimated. Root cause: limited training data at the tails (only 5 fish over 35").
- **All photos have a person**: No training data yet for fish-only photos.
- **Species imbalance**: 85% steelhead_traveler, 15% steelhead_holding. Unlabeled set shows 59/39 split — more holding photos with labels would help.
- **Hand detection rate**: Only 30% of photos pass the sanity filter. Will improve with higher-res photos and clearer finger visibility.

## What Would Help Most

1. **More labeled photos at the extremes** — fish under 27" and over 35"
2. **More steelhead_holding labels** — currently underrepresented
3. **Photos without a person** — to train the no-person path
4. **Lengths for the 111 unlabeled photos** — even approximate measurements add value

## Roadmap

- **Phase 2**: Deploy model to iOS app, capture full feature vectors in Supabase, instrument correction feedback loop
- **Phase 3**: Server-side endpoint for Android, automated retraining pipeline
- **Phase 4**: YOLO keypoint detection (nose-to-tail) for true length measurement
