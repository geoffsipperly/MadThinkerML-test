"""Export trained ViT sex classifier to CoreML .mlpackage format."""

import torch
import timm
import coremltools as ct

# IMPORTANT: must match ImageFolder order from training
CLASSES = ["female", "male"]
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
    weights_path: str = "models/vit_fish_sex.pt",
    out_path: str = "models/ViTFishSex.mlpackage",
):
    print("Loading trained PyTorch model from", weights_path)
    model = load_trained_model(weights_path)

    example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    print("Tracing model...")
    traced = torch.jit.trace(model, example_input)
    traced.eval()

    classifier_config = ct.ClassifierConfig(class_labels=CLASSES)

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input",
                shape=example_input.shape,
            )
        ],
        classifier_config=classifier_config,
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )

    print("Saving CoreML model to", out_path)
    mlmodel.save(out_path)
    print("Done. Saved:", out_path)


if __name__ == "__main__":
    export_coreml()
