"""Export trained ViT species classifier to CoreML .mlpackage format."""

import torch
import timm
import coremltools as ct

# IMPORTANT: must match ImageFolder alphabetical order from training
CLASSES = [
    "articchar_holding",
    "articchar_traveler",
    "atlantic_salmon",
    "brook_holding",
    "brook_trout",
    "brown_trout",
    "carp",
    "chinook_salmon",
    "chum_salmon",
    "coho_salmon",
    "cutthroat_trout",
    "grayling",
    "largemouth_bass",
    "muskellunge",
    "northern_pike",
    "pink_salmon",
    "rainbow_holding",
    "rainbow_lake",
    "rainbow_traveler",
    "sea_run_trout",
    "smallmouth_bass",
    "sockeye_salmon",
    "steelhead_holding",
    "steelhead_traveler",
]

NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
ARCH = "vit_tiny_patch16_224"


def load_trained_model(weights_path: str):
    print(f"Creating model {ARCH} with num_classes={NUM_CLASSES}")
    model = timm.create_model(
        ARCH,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    print("Loading weights from:", weights_path)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def export_coreml(
    weights_path: str = "models/vit_fish_species.pt",
    out_path: str = "models/ViTFishSpecies.mlpackage",
):
    model = load_trained_model(weights_path)

    example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    print("Tracing model...")
    traced = torch.jit.trace(model, example_input)
    traced.eval()

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="image",
                shape=example_input.shape,
            )
        ],
        minimum_deployment_target=ct.target.iOS16,
    )

    print("Saving CoreML model to", out_path)
    mlmodel.save(out_path)
    print("Done. Saved:", out_path)


if __name__ == "__main__":
    export_coreml()
