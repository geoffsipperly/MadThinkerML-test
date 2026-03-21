"""
Sea-Run Trout Image Downloader
================================
Downloads research-grade, openly licensed photos of sea-run trout
from iNaturalist for use in ML training datasets.

Species included:
  - Salmo trutta (Brown Trout / Sea Trout) — taxon 48028
  - Oncorhynchus mykiss (Rainbow Trout / Steelhead) — taxon 47516

Usage:
    pip install requests
    python download_sea_run_trout.py

Images are saved into a folder called `sea_run_trout_images/` with
a metadata CSV for tracking provenance and licenses.
"""

import csv
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Please install the requests library first:  pip install requests")
    sys.exit(1)

# ──────────────────────────── CONFIG ────────────────────────────
OUTPUT_DIR = Path("sea_run_trout_images")
METADATA_FILE = OUTPUT_DIR / "metadata.csv"
MAX_IMAGES = 100  # total images to download (across all species)
PER_PAGE = 50     # iNaturalist max per request is 200; 50 is polite

# iNaturalist taxon IDs
TAXA = {
    48028: "Salmo_trutta",        # Brown Trout / Sea Trout
    47516: "Oncorhynchus_mykiss", # Rainbow Trout / Steelhead
}

API_BASE = "https://api.inaturalist.org/v1"
HEADERS = {
    "User-Agent": "SeaRunTroutDatasetCollector/1.0 (ML research; contact your-email@example.com)"
}

# Only download images with these Creative Commons licenses
ALLOWED_LICENSES = {"cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa", "cc0"}


# ──────────────────────────── HELPERS ───────────────────────────
def fetch_observations(taxon_id: int, page: int = 1, per_page: int = PER_PAGE) -> dict:
    """Query iNaturalist for research-grade observations with photos."""
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "photos": True,
        "photo_license": "cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa,cc0",
        "per_page": per_page,
        "page": page,
        "order": "desc",
        "order_by": "votes",  # prefer highly-rated observations
    }
    resp = requests.get(f"{API_BASE}/observations", params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_image(url: str, dest: Path) -> bool:
    """Download a single image file."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ⚠ Failed to download {url}: {e}")
        return False


def photo_url_medium(photo: dict) -> str:
    """Get a medium-sized image URL from an iNaturalist photo object."""
    url = photo.get("url", "")
    # iNat returns square thumbnails by default; swap to medium size
    return url.replace("square", "medium")


# ──────────────────────────── MAIN ──────────────────────────────
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    downloaded = 0
    metadata_rows = []

    print(f"🐟 Sea-Run Trout Image Downloader")
    print(f"   Target: {MAX_IMAGES} images")
    print(f"   Output: {OUTPUT_DIR.resolve()}\n")

    images_per_taxon = MAX_IMAGES // len(TAXA)

    for taxon_id, species_label in TAXA.items():
        species_dir = OUTPUT_DIR / species_label
        species_dir.mkdir(exist_ok=True)

        print(f"── {species_label} (taxon {taxon_id}) ──")
        species_count = 0
        page = 1

        while species_count < images_per_taxon:
            print(f"  Fetching page {page}...")
            data = fetch_observations(taxon_id, page=page, per_page=PER_PAGE)
            results = data.get("results", [])

            if not results:
                print(f"  No more observations found.")
                break

            for obs in results:
                if species_count >= images_per_taxon:
                    break

                photos = obs.get("photos", [])
                if not photos:
                    continue

                # Take only the first (best) photo per observation
                photo = photos[0]
                license_code = (photo.get("license_code") or "").lower()

                if license_code not in ALLOWED_LICENSES:
                    continue

                url = photo_url_medium(photo)
                ext = "jpg"
                filename = f"{species_label}_{obs['id']}_{photo['id']}.{ext}"
                dest = species_dir / filename

                if dest.exists():
                    species_count += 1
                    continue

                print(f"  [{downloaded + 1}/{MAX_IMAGES}] Downloading {filename}")
                if download_image(url, dest):
                    species_count += 1
                    downloaded += 1
                    metadata_rows.append({
                        "filename": str(species_dir / filename),
                        "observation_id": obs["id"],
                        "photo_id": photo["id"],
                        "species": obs.get("taxon", {}).get("name", species_label),
                        "common_name": obs.get("taxon", {}).get("preferred_common_name", ""),
                        "license": license_code,
                        "url": url,
                        "observer": obs.get("user", {}).get("login", ""),
                        "observed_on": obs.get("observed_on", ""),
                        "location": obs.get("place_guess", ""),
                    })

                # Be polite: ~1 request/sec
                time.sleep(1)

            page += 1
            time.sleep(1)  # pause between pages

        print(f"  ✓ {species_count} images downloaded for {species_label}\n")

    # Write metadata CSV
    if metadata_rows:
        with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)
        print(f"📄 Metadata saved to {METADATA_FILE}")

    print(f"\n✅ Done! {downloaded} images downloaded to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
