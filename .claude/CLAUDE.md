# MadThinkerML - ML Training Pipeline

## Project Overview
Python ML training pipeline for EmeraldWatersAnglers iOS app. Trains species classification (ViT), sex classification (ViT), and length estimation (gradient-boosted regressor) models, then exports to CoreML for on-device inference.

## Key Scripts
- `scripts/train_vit_species.py` — Trains ViT species classifier using ImageFolder format from `data/fish_species/`
- `scripts/train_vit_sex.py` — Trains ViT sex classifier
- `scripts/export_vit_species_coreml.py` — Exports trained ViT to CoreML `.mlpackage` using `torch.export` + coremltools 9
- `scripts/split_train_val.py` — Splits downloaded images 80/20 into train/val

## Current Species Model (as of 2026-03-21)
- **Architecture**: `vit_tiny_patch16_224` (timm)
- **Classes**: 9 (original set) — articchar_holding, articchar_traveler, brook_holding, grayling, rainbow_holding, rainbow_lake, rainbow_traveler, steelhead_holding, steelhead_traveler
- **Val accuracy**: 87.6%
- **Empty class folders preserved** in `data/fish_species/` for 15 additional species (atlantic_salmon, brook_trout, brown_trout, carp, chinook_salmon, chum_salmon, coho_salmon, cutthroat_trout, largemouth_bass, muskellunge, northern_pike, pink_salmon, sea_run_trout, smallmouth_bass, sockeye_salmon) — images were downloaded but removed due to low quality. Need curated images before retraining.

## CoreML Export Notes
- **Must use `torch.export`** — `jit.trace` fails with timm ViT attention ops in coremltools
- **Must specify input name `"image"` and output name `"logits"`** — iOS code expects these exact names via `ViTInputFeatureProvider`
- Requires `exported.run_decompositions({})` before `ct.convert`
- coremltools 9.0 + torch 2.7.0 is the tested working combination

## Environment
- Python 3.13, torch 2.7.0, timm, coremltools 9.0
- `pip3 install torch==2.7.0 torchvision==0.22.0 timm coremltools requests`
- Model weights saved to `models/vit_fish_species.pt` (gitignored)
- CoreML output: `models/ViTFishSpecies.mlpackage` (gitignored, copy to iOS project manually)

## Data Sources
- Original 9 classes: curated photos from guides (high quality)
- New species: downloaded from iNaturalist API (`api.inaturalist.org/v1/observations`). Note: original download scripts on Desktop had wrong taxon IDs. Correct IDs found via taxa search endpoint.

## Retraining Workflow
1. Add curated images to `data/fish_species/train/<class>/` and `data/fish_species/val/<class>/`
2. Remove empty class folders temporarily (ImageFolder fails on empty dirs): move to `.empty_classes/`
3. Run `python3 scripts/train_vit_species.py`
4. Restore empty folders
5. Run `python3 scripts/export_vit_species_coreml.py`
6. Copy `models/ViTFishSpecies.mlpackage` to iOS project at `SkeenaSystem/Models/`
7. Update `speciesLabels` array in `CatchPhotoAnalyzer.swift` to match alphabetical class order
8. Update `speciesDisplayNames` in `CatchChatViewModel.swift` if new class names need mapping
