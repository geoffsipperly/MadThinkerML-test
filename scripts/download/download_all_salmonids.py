"""
All Salmonids Image Downloader
================================
Downloads research-grade, openly licensed photos of ALL 8 species
from iNaturalist in a single run. Combines the sea-run trout and
salmon species into one unified dataset.

Species included:
  - Salmo trutta (Brown Trout / Sea Trout) — taxon 48028
  - Oncorhynchus mykiss (Steelhead / Rainbow Trout) — taxon 47516
  - Oncorhynchus tshawytscha (Chinook Salmon) — taxon 48222
  - Salmo salar (Atlantic Salmon) — taxon 48221
  - Oncorhynchus kisutch (Coho Salmon) — taxon 48233
  - Oncorhynchus keta (Chum Salmon) — taxon 48236
  - Oncorhynchus gorbuscha (Pink Salmon) — taxon 48234
  - Oncorhynchus nerka (Sockeye Salmon) — taxon 48231

Usage:
    pip install requests
    python download_all_salmonids.py

All images are saved into `salmonid_dataset/` with subfolders per
species and a combined metadata CSV for the full dataset.
"""

import csv, os, sys, time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Please install the requests library first:  pip install requests")
    sys.exit(1)

# ──────────────────────────── CONFIG ────────────────────────────
OUTPUT_DIR = Path("salmonid_dataset")
METADATA_FILE = OUTPUT_DIR / "metadata.csv"
IMAGES_PER_SPECIES = 100   # images to download per species
PER_PAGE = 50              # results per API request

# All 8 species: iNaturalist taxon ID → folder label
TAXA = {
    48028: "Salmo_trutta",               # Brown Trout / Sea Trout
    47516: "Oncorhynchus_mykiss",        # Steelhead / Rainbow Trout
    48222: "Oncorhynchus_tshawytscha",   # Chinook Salmon
    48221: "Salmo_salar",                # Atlantic Salmon
    48233: "Oncorhynchus_kisutch",       # Coho Salmon
    48236: "Oncorhynchus_keta",          # Chum Salmon
    48234: "Oncorhynchus_gorbuscha",     # Pink Salmon
    48231: "Oncorhynchus_nerka",         # Sockeye Salmon
}

API_BASE = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "SalmonidDatasetCollector/1.0 (ML research)"}
ALLOWED_LICENSES = {"cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa", "cc0"}


# ──────────────────────────── HELPERS ───────────────────────────
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
        print(f"    ⚠ Failed: {e}")
        return False


def photo_url_medium(photo):
    return photo.get("url", "").replace("square", "medium")


# ──────────────────────────── MAIN ──────────────────────────────
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    total_downloaded = 0
    metadata_rows = []
    total_target = IMAGES_PER_SPECIES * len(TAXA)

    print(f"🐟 Salmonid Dataset Downloader")
    print(f"   {len(TAXA)} species × {IMAGES_PER_SPECIES} images = {total_target} target images")
    print(f"   Output: {OUTPUT_DIR.resolve()}\n")

    for taxon_id, label in TAXA.items():
        species_dir = OUTPUT_DIR / label
        species_dir.mkdir(exist_ok=True)

        print(f"━━ {label} (taxon {taxon_id}) ━━")
        count, page = 0, 1

        while count < IMAGES_PER_SPECIES:
            print(f"  Fetching page {page}...")
            try:
                data = fetch_observations(taxon_id, page=page)
            except Exception as e:
                print(f"  ⚠ API error: {e}. Skipping to next species.")
                break

            results = data.get("results", [])
            if not results:
                print(f"  No more observations available.")
                break

            for obs in results:
                if count >= IMAGES_PER_SPECIES:
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

                print(f"  [{total_downloaded+1}/{total_target}] {fname}")
                if download_image(url, dest):
                    count += 1
                    total_downloaded += 1
                    metadata_rows.append({
                        "filename": str(dest),
                        "observation_id": obs["id"],
                        "photo_id": photo["id"],
                        "species": obs.get("taxon", {}).get("name", label),
                        "common_name": obs.get("taxon", {}).get("preferred_common_name", ""),
                        "license": lc,
                        "url": url,
                        "observer": obs.get("user", {}).get("login", ""),
                        "observed_on": obs.get("observed_on", ""),
                        "location": obs.get("place_guess", ""),
                    })

                time.sleep(1)

            page += 1
            time.sleep(1)

        print(f"  ✓ {count} images downloaded for {label}\n")

    # Write combined metadata CSV
    if metadata_rows:
        with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)
        print(f"📄 Metadata saved to {METADATA_FILE}")

    # Summary
    print(f"\n{'='*50}")
    print(f"✅ COMPLETE: {total_downloaded} images across {len(TAXA)} species")
    print(f"   Dataset: {OUTPUT_DIR.resolve()}")
    print(f"   Metadata: {METADATA_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
