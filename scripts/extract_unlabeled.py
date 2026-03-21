"""
Extract features from unlabeled photos for distribution analysis.

Runs YOLO + ViT on each photo (same as extract_features.py) but doesn't
require ground truth lengths. Used to compare feature distributions between
labeled and unlabeled sets.

Usage:
    python scripts/extract_unlabeled.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Reuse constants and functions from extract_features
PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNLABELED_DIR = PROJECT_ROOT / "data" / "unlabeled" / "images"
OUTPUT_DIR = PROJECT_ROOT / "output"

YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "yolo_fish_detector.pt"
VIT_MODEL_PATH = PROJECT_ROOT / "models" / "vit_fish_species.pt"

IMG_SIZE = 640.0
FISH_CLASS = 0
PERSON_CLASS = 1
MIN_PRIMARY_CONFIDENCE = 0.08
MIN_ASPECT = 1.0
MAX_HEIGHT_FRACTION = 0.9
MAX_AREA_FRACTION = 0.8
FALLBACK_MIN_CONFIDENCE = 0.01
PIXELS_PER_INCH = 11.7

VIT_ARCH = "vit_tiny_patch16_224"
VIT_IMG_SIZE = 224
SPECIES_CLASSES = [
    "articchar_holding", "articchar_traveler", "brook_holding",
    "grayling", "rainbow_holding", "rainbow_lake",
    "rainbow_traveler", "steelhead_holding", "steelhead_traveler",
]

# Import the core functions from extract_features
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from extract_features import (
    load_yolo_model, load_vit_model, run_yolo, run_vit,
    compute_features, baseline_prediction,
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted([
        f for f in UNLABELED_DIR.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    print(f"Found {len(images)} unlabeled images in {UNLABELED_DIR}")

    if not images:
        sys.exit("No images found.")

    yolo_model = load_yolo_model()
    vit_model, vit_tfms, vit_device = load_vit_model()

    rows = []
    skipped = 0

    for image_path in images:
        fish_box, person_box, orig_w, orig_h = run_yolo(yolo_model, image_path)
        species_idx, species_conf = run_vit(vit_model, vit_tfms, vit_device, image_path)
        features = compute_features(fish_box, person_box, species_idx, species_conf, orig_w, orig_h)

        if features is None:
            print(f"  MISS {image_path.name}")
            skipped += 1
            continue

        baseline = baseline_prediction(fish_box)
        features["filename"] = image_path.name
        features["baseline_prediction"] = baseline
        features["species_name"] = SPECIES_CLASSES[species_idx]
        features["person_in_photo"] = "yes" if person_box else "no"
        rows.append(features)

        person_flag = "P" if person_box else " "
        print(f"  OK   {image_path.name:<40} baseline={baseline:5.1f}\"  {person_flag}  {SPECIES_CLASSES[species_idx]}")

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "unlabeled_features.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"UNLABELED EXTRACTION COMPLETE")
    print(f"  Processed: {len(rows)}")
    print(f"  Skipped: {skipped}")
    print(f"  With person: {sum(1 for r in rows if r['person_detected'] == 1.0)}")
    print(f"  Without person: {sum(1 for r in rows if r['person_detected'] == 0.0)}")
    print(f"  Saved to: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
