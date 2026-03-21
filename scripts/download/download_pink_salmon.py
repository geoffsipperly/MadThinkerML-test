"""
Pink Salmon Image Downloader
================================
Downloads research-grade, openly licensed photos of Pink Salmon
(Oncorhynchus gorbuscha) from iNaturalist for ML training datasets.

Usage:
    pip install requests
    python download_pink_salmon.py
"""

import csv, os, sys, time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Please install the requests library first:  pip install requests")
    sys.exit(1)

OUTPUT_DIR = Path("pink_salmon_images")
METADATA_FILE = OUTPUT_DIR / "metadata.csv"
MAX_IMAGES = 100
PER_PAGE = 50

TAXA = {
    48234: "Oncorhynchus_gorbuscha",  # Pink Salmon
}

API_BASE = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "SalmonDatasetCollector/1.0 (ML research)"}
ALLOWED_LICENSES = {"cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa", "cc0"}


def fetch_observations(taxon_id, page=1, per_page=PER_PAGE):
    params = {
        "taxon_id": taxon_id, "quality_grade": "research", "photos": True,
        "photo_license": "cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa,cc0",
        "per_page": per_page, "page": page, "order": "desc", "order_by": "votes",
    }
    resp = requests.get(f"{API_BASE}/observations", params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_image(url, dest):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ⚠ Failed: {e}")
        return False


def photo_url_medium(photo):
    return photo.get("url", "").replace("square", "medium")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    downloaded = 0
    metadata_rows = []

    print(f"🐟 Pink Salmon Image Downloader")
    print(f"   Target: {MAX_IMAGES} images → {OUTPUT_DIR.resolve()}\n")

    for taxon_id, label in TAXA.items():
        species_dir = OUTPUT_DIR / label
        species_dir.mkdir(exist_ok=True)
        print(f"── {label} (taxon {taxon_id}) ──")
        count, page = 0, 1

        while count < MAX_IMAGES:
            print(f"  Fetching page {page}...")
            data = fetch_observations(taxon_id, page=page)
            results = data.get("results", [])
            if not results:
                break

            for obs in results:
                if count >= MAX_IMAGES:
                    break
                photos = obs.get("photos", [])
                if not photos:
                    continue
                photo = photos[0]
                lc = (photo.get("license_code") or "").lower()
                if lc not in ALLOWED_LICENSES:
                    continue

                url = photo_url_medium(photo)
                fname = f"{label}_{obs['id']}_{photo['id']}.jpg"
                dest = species_dir / fname
                if dest.exists():
                    count += 1
                    continue

                print(f"  [{downloaded+1}/{MAX_IMAGES}] {fname}")
                if download_image(url, dest):
                    count += 1
                    downloaded += 1
                    metadata_rows.append({
                        "filename": str(dest), "observation_id": obs["id"],
                        "photo_id": photo["id"],
                        "species": obs.get("taxon", {}).get("name", label),
                        "common_name": obs.get("taxon", {}).get("preferred_common_name", ""),
                        "license": lc, "url": url,
                        "observer": obs.get("user", {}).get("login", ""),
                        "observed_on": obs.get("observed_on", ""),
                        "location": obs.get("place_guess", ""),
                    })
                time.sleep(1)
            page += 1
            time.sleep(1)

        print(f"  ✓ {count} images\n")

    if metadata_rows:
        with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)
        print(f"📄 Metadata → {METADATA_FILE}")

    print(f"\n✅ Done! {downloaded} images → {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
