"""
Split downloaded salmonid images into train/val and merge into fish_species dataset.

Reads from salmonid_dataset/<scientific_name>/ and copies into
data/fish_species/{train,val}/<common_name>/ with an 80/20 split.

Skips Oncorhynchus_mykiss (steelhead/rainbow) since those are already
in the dataset.

Usage:
    python scripts/split_new_species.py
"""

import random
import shutil
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "salmonid_dataset"
DEST_DIR = PROJECT_ROOT / "data" / "fish_species"

# Map scientific subdirectory names to training class names
SPECIES_MAP = {
    "Oncorhynchus_tshawytscha": "chinook_salmon",
    "Salmo_salar": "atlantic_salmon",
    "Oncorhynchus_kisutch": "coho_salmon",
    "Oncorhynchus_keta": "chum_salmon",
    "Oncorhynchus_gorbuscha": "pink_salmon",
    "Oncorhynchus_nerka": "sockeye_salmon",
    "Salmo_trutta": "sea_run_trout",
    "Oncorhynchus_clarkii": "cutthroat_trout",
    "Salvelinus_fontinalis": "brook_trout",
    "Salmo_trutta_brown": "brown_trout",
}

# Skip — already in the dataset
SKIP = {"Oncorhynchus_mykiss"}

TRAIN_RATIO = 0.8


def main():
    if not SOURCE_DIR.exists():
        print(f"Source directory not found: {SOURCE_DIR}")
        print("Run download_all_salmonids.py first.")
        return

    for scientific_name in sorted(SOURCE_DIR.iterdir()):
        if not scientific_name.is_dir():
            continue

        dirname = scientific_name.name
        if dirname in SKIP:
            print(f"Skipping {dirname} (already in dataset)")
            continue

        if dirname not in SPECIES_MAP:
            print(f"Skipping {dirname} (not in species map)")
            continue

        class_name = SPECIES_MAP[dirname]
        images = sorted([f for f in scientific_name.iterdir()
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

        if not images:
            print(f"No images found for {dirname}")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create destination directories
        train_dir = DEST_DIR / "train" / class_name
        val_dir = DEST_DIR / "val" / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img in train_images:
            shutil.copy2(img, train_dir / img.name)
        for img in val_images:
            shutil.copy2(img, val_dir / img.name)

        print(f"{class_name}: {len(train_images)} train / {len(val_images)} val "
              f"(from {len(images)} total)")

    # Print final summary
    print("\nFinal dataset summary:")
    for split in ["train", "val"]:
        split_dir = DEST_DIR / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                count = len(list(cls_dir.glob("*.[jJ][pP][gG]")) +
                           list(cls_dir.glob("*.[pP][nN][gG]")))
                print(f"  {split}/{cls_dir.name}: {count} images")


if __name__ == "__main__":
    main()
