"""
Extract features from ground truth fish photos for length regression.

Runs YOLO (fish + person detection) and ViT (species classification) on each
photo, then computes 14 features per image for the XGBoost length regressor.

Usage:
    python scripts/extract_features.py

Expects:
    data/ground_truth/images/   — fish photos (jpg/png)
    data/ground_truth/labels.csv — columns: filename, length_inches, species (optional)

Outputs:
    output/features.csv — 14 features + ground truth + baseline prediction per image
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

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPICWATERS_ML = Path.home() / "dev" / "EpicWatersML"

IMAGES_DIR = PROJECT_ROOT / "data" / "ground_truth" / "images"
LABELS_CSV = PROJECT_ROOT / "data" / "ground_truth" / "labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

YOLO_MODEL_PATH = EPICWATERS_ML / "runs" / "fish_det" / "steelhead_fish_v3" / "weights" / "best.pt"
VIT_MODEL_PATH = EPICWATERS_ML / "vit_fish_species_tiny_best.pt"

# ── Constants (matching Swift CatchPhotoAnalyzer) ────────────────────────────
IMG_SIZE = 640.0
FISH_CLASS = 0
PERSON_CLASS = 1
MIN_PRIMARY_CONFIDENCE = 0.08
MIN_ASPECT = 1.0
MAX_HEIGHT_FRACTION = 0.9
MAX_AREA_FRACTION = 0.8
FALLBACK_MIN_CONFIDENCE = 0.01
PIXELS_PER_INCH = 11.7  # Current heuristic constant

# ── ViT Species Config ───────────────────────────────────────────────────────
VIT_ARCH = "vit_tiny_patch16_224"
VIT_IMG_SIZE = 224
SPECIES_CLASSES = [
    "articchar_holding", "articchar_traveler", "brook_holding",
    "grayling", "rainbow_holding", "rainbow_lake",
    "rainbow_traveler", "steelhead_holding", "steelhead_traveler",
]


def load_yolo_model():
    print(f"Loading YOLO model from {YOLO_MODEL_PATH}")
    if not YOLO_MODEL_PATH.exists():
        sys.exit(f"YOLO model not found: {YOLO_MODEL_PATH}")
    return YOLO(str(YOLO_MODEL_PATH))


def load_vit_model():
    print(f"Loading ViT species model from {VIT_MODEL_PATH}")
    if not VIT_MODEL_PATH.exists():
        sys.exit(f"ViT model not found: {VIT_MODEL_PATH}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model(VIT_ARCH, pretrained=False, num_classes=len(SPECIES_CLASSES))
    model.load_state_dict(torch.load(str(VIT_MODEL_PATH), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize((VIT_IMG_SIZE, VIT_IMG_SIZE)),
        transforms.CenterCrop(VIT_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return model, tfms, device


def run_yolo(yolo_model, image_path):
    """Run YOLO and extract fish + person bounding boxes in 640x640 space."""
    img = Image.open(image_path)
    orig_w, orig_h = img.size
    scale_x = IMG_SIZE / orig_w
    scale_y = IMG_SIZE / orig_h

    results = yolo_model.predict(source=str(image_path), imgsz=640, conf=0.001, verbose=False)

    fish_box = None
    fish_conf = 0.0
    person_box = None
    person_conf = 0.0

    if len(results) == 0 or len(results[0].boxes) == 0:
        return None, None, orig_w, orig_h

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xywh = box.xywh[0].cpu().numpy()

        # Scale to 640x640 space
        x_c = xywh[0] * scale_x
        y_c = xywh[1] * scale_y
        w = xywh[2] * scale_x
        h = xywh[3] * scale_y

        if cls == FISH_CLASS:
            # Apply geometry filters (primary path)
            aspect = w / max(h, 1.0)
            h_frac = h / IMG_SIZE
            area_frac = (w * h) / (IMG_SIZE * IMG_SIZE)

            passes_filters = (
                conf >= MIN_PRIMARY_CONFIDENCE
                and aspect >= MIN_ASPECT
                and h_frac <= MAX_HEIGHT_FRACTION
                and area_frac <= MAX_AREA_FRACTION
            )

            if passes_filters and conf > fish_conf:
                fish_conf = conf
                fish_box = {"x_center": x_c, "y_center": y_c, "width": w, "height": h, "conf": conf}
            elif fish_box is None and conf >= FALLBACK_MIN_CONFIDENCE and conf > fish_conf:
                # Fallback: best fish regardless of geometry
                fish_conf = conf
                fish_box = {"x_center": x_c, "y_center": y_c, "width": w, "height": h, "conf": conf}

        elif cls == PERSON_CLASS and conf > person_conf:
            person_conf = conf
            person_box = {"x_center": x_c, "y_center": y_c, "width": w, "height": h, "conf": conf}

    return fish_box, person_box, orig_w, orig_h


def run_vit(vit_model, tfms, device, image_path):
    """Run ViT species classifier. Returns (species_index, confidence)."""
    img = Image.open(image_path).convert("RGB")
    tensor = tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = vit_model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    return int(idx.item()), float(conf.item())


def compute_features(fish_box, person_box, species_idx, species_conf, orig_w, orig_h):
    """Compute the 14-feature vector."""
    if fish_box is None:
        return None

    fw = fish_box["width"]
    fh = fish_box["height"]

    # Fish geometry
    fish_box_width = fw
    fish_box_height = fh
    fish_box_area = fw * fh
    fish_aspect_ratio = fw / max(fh, 1.0)
    fish_box_x_center = fish_box["x_center"]
    fish_box_y_center = fish_box["y_center"]
    fish_confidence = fish_box["conf"]

    # Person reference
    person_detected = 1.0 if person_box is not None else 0.0
    person_box_height = person_box["height"] if person_box else 0.0
    fish_to_person_ratio = (max(fw, fh) / person_box["height"]) if person_box else 0.0

    # Species
    species_index = float(species_idx)
    species_confidence_val = species_conf

    # Image geometry
    image_aspect_ratio = orig_w / max(orig_h, 1)
    diagonal = np.sqrt(fw ** 2 + fh ** 2)
    frame_diagonal = np.sqrt(IMG_SIZE ** 2 + IMG_SIZE ** 2)
    diagonal_fraction = diagonal / frame_diagonal

    return {
        "fish_box_width": fish_box_width,
        "fish_box_height": fish_box_height,
        "fish_box_area": fish_box_area,
        "fish_aspect_ratio": fish_aspect_ratio,
        "fish_box_x_center": fish_box_x_center,
        "fish_box_y_center": fish_box_y_center,
        "fish_confidence": fish_confidence,
        "person_detected": person_detected,
        "person_box_height": person_box_height,
        "fish_to_person_ratio": fish_to_person_ratio,
        "species_index": species_index,
        "species_confidence": species_confidence_val,
        "image_aspect_ratio": image_aspect_ratio,
        "diagonal_fraction": diagonal_fraction,
    }


def baseline_prediction(fish_box):
    """Current heuristic: max(w,h) / pixelsPerInch, clamped to [10, 47]."""
    if fish_box is None:
        return None
    pixel_length = max(fish_box["width"], fish_box["height"])
    raw = pixel_length / PIXELS_PER_INCH
    return float(np.clip(raw, 10.0, 47.0))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load labels — support both simple format and delta-report format
    if not LABELS_CSV.exists():
        sys.exit(
            f"Labels CSV not found: {LABELS_CSV}\n"
            f"Please create it with columns: filename, length_inches, species (optional)\n"
            f"  OR delta-report format: Image Filename, Final Length, Final Species"
        )

    labels_df = pd.read_csv(LABELS_CSV)

    # Normalize column names: support delta-report format
    col_map = {}
    if "Image Filename" in labels_df.columns:
        col_map["Image Filename"] = "filename"
    if "Final Length" in labels_df.columns:
        col_map["Final Length"] = "length_inches"
    if "Final Species" in labels_df.columns:
        col_map["Final Species"] = "species"
    if "Initial Length" in labels_df.columns:
        col_map["Initial Length"] = "initial_length"
    if col_map:
        labels_df = labels_df.rename(columns=col_map)

    required_cols = {"filename", "length_inches"}
    if not required_cols.issubset(labels_df.columns):
        sys.exit(f"Labels CSV must have columns: {required_cols}. Found: {set(labels_df.columns)}")

    # Convert length to numeric, drop non-numeric
    labels_df["length_inches"] = pd.to_numeric(labels_df["length_inches"], errors="coerce")

    print(f"Loaded {len(labels_df)} labeled images from {LABELS_CSV}")

    # Load models
    yolo_model = load_yolo_model()
    vit_model, vit_tfms, vit_device = load_vit_model()

    rows = []
    skipped = 0

    for _, row in labels_df.iterrows():
        filename = row["filename"]
        length_inches = row["length_inches"]

        if pd.isna(filename) or str(filename).strip() == "":
            skipped += 1
            continue

        filename = str(filename).strip()
        image_path = IMAGES_DIR / filename

        if not image_path.exists():
            print(f"  SKIP {filename} — image not found")
            skipped += 1
            continue

        if pd.isna(length_inches):
            print(f"  SKIP {filename} — no ground truth length")
            skipped += 1
            continue

        # Run YOLO
        fish_box, person_box, orig_w, orig_h = run_yolo(yolo_model, image_path)

        # Run ViT
        species_idx, species_conf = run_vit(vit_model, vit_tfms, vit_device, image_path)

        # Compute features
        features = compute_features(fish_box, person_box, species_idx, species_conf, orig_w, orig_h)

        if features is None:
            print(f"  MISS {filename} — no fish detected")
            skipped += 1
            continue

        # Baseline heuristic prediction
        baseline = baseline_prediction(fish_box)

        features["filename"] = filename
        features["length_inches"] = float(length_inches)
        features["baseline_prediction"] = baseline
        features["species_name"] = SPECIES_CLASSES[species_idx]
        features["person_in_photo"] = "yes" if person_box else "no"

        rows.append(features)

        baseline_err = abs(baseline - length_inches) if baseline else None
        person_flag = "P" if person_box else " "
        print(
            f"  OK   {filename:<40} "
            f"actual={length_inches:5.1f}\"  baseline={baseline:5.1f}\"  "
            f"err={baseline_err:+5.1f}\"  {person_flag}  {SPECIES_CLASSES[species_idx]}"
        )

    # Save
    features_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "features.csv"
    features_df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"  Images processed: {len(rows)}")
    print(f"  Skipped: {skipped}")
    print(f"  With person: {sum(1 for r in rows if r['person_detected'] == 1.0)}")
    print(f"  Without person: {sum(1 for r in rows if r['person_detected'] == 0.0)}")

    if rows:
        baseline_errors = [abs(r["baseline_prediction"] - r["length_inches"]) for r in rows]
        print(f"  Baseline MAE: {np.mean(baseline_errors):.2f}\"")

    print(f"\n  Features saved to: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
