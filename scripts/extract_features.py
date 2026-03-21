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
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGES_DIR = PROJECT_ROOT / "data" / "ground_truth" / "images"
LABELS_CSV = PROJECT_ROOT / "data" / "ground_truth" / "labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "yolo_fish_detector.pt"
VIT_MODEL_PATH = PROJECT_ROOT / "models" / "vit_fish_species.pt"
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

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


def load_hand_detector():
    print(f"Loading MediaPipe hand landmarker from {HAND_MODEL_PATH}")
    if not HAND_MODEL_PATH.exists():
        sys.exit(f"Hand landmarker model not found: {HAND_MODEL_PATH}")
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(options)


def run_hand_detection(hand_detector, image_path):
    """Detect hands and compute finger measurements in original image pixel space.

    Uses index-to-middle finger knuckle spacing as the primary width reference
    (~0.85" real-world for adults) and index finger MCP-to-TIP as the length
    reference (~3.5" real-world).

    Applies sanity filters to reject bad detections:
    - Finger width/length ratio must be plausible (length ~3-6x width)
    - PPI from finger must be within reasonable range for the image size
    - Rejects landmarks that are likely misidentified (e.g., on fish body)

    Returns dict with hand features, or None if no hands detected or all fail sanity.
    """
    mp_image = mp.Image.create_from_file(str(image_path))
    result = hand_detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    w, h = mp_image.width, mp_image.height
    best = None
    best_confidence = 0.0

    for i, hand in enumerate(result.hand_landmarks):
        # Key landmarks
        idx_mcp = hand[5]   # Index finger knuckle
        idx_pip = hand[6]   # Index finger first joint
        idx_tip = hand[8]   # Index fingertip
        mid_mcp = hand[9]   # Middle finger knuckle

        # Finger width: distance between index and middle knuckles (pixels)
        finger_width_px = np.sqrt(
            ((idx_mcp.x - mid_mcp.x) * w) ** 2 +
            ((idx_mcp.y - mid_mcp.y) * h) ** 2
        )

        # Index finger length: knuckle to tip (pixels)
        finger_length_px = np.sqrt(
            ((idx_mcp.x - idx_tip.x) * w) ** 2 +
            ((idx_mcp.y - idx_tip.y) * h) ** 2
        )

        # Use confidence from handedness
        conf = result.handedness[i][0].score if result.handedness[i] else 0.5

        # ── Sanity filters ───────────────────────────────────────────
        # Min pixel size: finger landmarks too small = likely bad detection
        if finger_width_px < 10 or finger_length_px < 15:
            continue

        # Finger length should be ~3-6x finger width for a real hand.
        # Outside 1.5-10x is likely a misdetection (landmarks on fish, etc.)
        length_to_width = finger_length_px / max(finger_width_px, 1)
        if length_to_width < 1.5 or length_to_width > 10.0:
            continue

        # PPI sanity: real-world finger width ~0.85" implies PPI.
        # For a typical photo (2000-5000px wide), PPI should be ~20-200.
        # Anything outside 10-500 is unreasonable.
        ppi = finger_width_px / 0.85
        if ppi < 10 or ppi > 500:
            continue

        # Implied fish length sanity: use PPI to estimate fish length.
        # If the image diagonal is D pixels, max plausible fish length
        # is D/ppi inches. If that's < 5" or > 80", PPI is wrong.
        img_diag = np.sqrt(w ** 2 + h ** 2)
        max_fish_inches = img_diag / ppi
        if max_fish_inches < 5 or max_fish_inches > 80:
            continue

        if conf > best_confidence:
            best_confidence = conf
            best = {
                "finger_width_px": finger_width_px,
                "finger_length_px": finger_length_px,
                "hand_confidence": conf,
                "pixels_per_inch_from_finger": ppi,
            }

    return best


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


def compute_features(fish_box, person_box, species_idx, species_conf, orig_w, orig_h, hand_result=None):
    """Compute feature vector from detection results."""
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
    person_box_width = person_box["width"] if person_box else 0.0
    person_aspect_ratio = (person_box["width"] / max(person_box["height"], 1.0)) if person_box else 0.0
    fish_to_person_ratio = (max(fw, fh) / person_box["height"]) if person_box else 0.0

    # Species
    species_index = float(species_idx)
    species_confidence_val = species_conf

    # Image geometry
    image_aspect_ratio = orig_w / max(orig_h, 1)
    diagonal = np.sqrt(fw ** 2 + fh ** 2)
    frame_diagonal = np.sqrt(IMG_SIZE ** 2 + IMG_SIZE ** 2)
    diagonal_fraction = diagonal / frame_diagonal

    # Hand/finger reference (in original image pixel space)
    hand_detected = 1.0 if hand_result is not None else 0.0
    finger_width_px = hand_result["finger_width_px"] if hand_result else 0.0
    finger_length_px = hand_result["finger_length_px"] if hand_result else 0.0
    ppi_from_finger = hand_result["pixels_per_inch_from_finger"] if hand_result else 0.0

    # Fish pixel length in original image space (for finger-based ratio)
    fish_pixel_orig = max(fw / (IMG_SIZE / orig_w), fh / (IMG_SIZE / orig_h))
    fish_to_finger_width = (fish_pixel_orig / finger_width_px) if finger_width_px > 0 else 0.0
    fish_to_finger_length = (fish_pixel_orig / finger_length_px) if finger_length_px > 0 else 0.0
    # Direct length estimate from finger PPI
    fish_inches_from_finger = (fish_pixel_orig / ppi_from_finger) if ppi_from_finger > 0 else 0.0

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
        "person_box_width": person_box_width,
        "person_aspect_ratio": person_aspect_ratio,
        "fish_to_person_ratio": fish_to_person_ratio,
        "species_index": species_index,
        "species_confidence": species_confidence_val,
        "image_aspect_ratio": image_aspect_ratio,
        "diagonal_fraction": diagonal_fraction,
        "hand_detected": hand_detected,
        "finger_width_px": finger_width_px,
        "finger_length_px": finger_length_px,
        "ppi_from_finger": ppi_from_finger,
        "fish_to_finger_width": fish_to_finger_width,
        "fish_to_finger_length": fish_to_finger_length,
        "fish_inches_from_finger": fish_inches_from_finger,
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
    hand_detector = load_hand_detector()

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

        # Run hand detection
        hand_result = run_hand_detection(hand_detector, image_path)

        # Compute features
        features = compute_features(fish_box, person_box, species_idx, species_conf, orig_w, orig_h, hand_result)

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
        features["hand_in_photo"] = "yes" if hand_result else "no"

        rows.append(features)

        baseline_err = abs(baseline - length_inches) if baseline else None
        person_flag = "P" if person_box else " "
        hand_flag = "H" if hand_result else " "
        finger_est = f"finger={features['fish_inches_from_finger']:5.1f}\"" if hand_result else "finger=  N/A"
        print(
            f"  OK   {filename:<40} "
            f"actual={length_inches:5.1f}\"  baseline={baseline:5.1f}\"  "
            f"err={baseline_err:+5.1f}\"  {person_flag}{hand_flag}  {finger_est}  {SPECIES_CLASSES[species_idx]}"
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
    print(f"  With hand: {sum(1 for r in rows if r['hand_detected'] == 1.0)}")
    print(f"  Without hand: {sum(1 for r in rows if r['hand_detected'] == 0.0)}")

    if rows:
        baseline_errors = [abs(r["baseline_prediction"] - r["length_inches"]) for r in rows]
        print(f"  Baseline MAE: {np.mean(baseline_errors):.2f}\"")

        # Finger-based estimate MAE (only for photos with hands)
        hand_rows = [r for r in rows if r["hand_detected"] == 1.0 and r["fish_inches_from_finger"] > 0]
        if hand_rows:
            finger_errors = [abs(r["fish_inches_from_finger"] - r["length_inches"]) for r in hand_rows]
            print(f"  Finger-based MAE: {np.mean(finger_errors):.2f}\" (n={len(hand_rows)})")

    print(f"\n  Features saved to: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
