"""Train ViT-Tiny species classifier on fish_species ImageFolder dataset."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# Paths (relative to repo root — run from MadThinkerML/)
DATA_ROOT = Path("data/fish_species")
CHECKPOINT_PATH = Path("models/vit_fish_species.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparams
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 3e-5
IMG_SIZE = 224
ARCH = "vit_tiny_patch16_224"

CLASSES = None


def get_dataloaders(root: Path):
    global CLASSES

    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    train_ds = datasets.ImageFolder(root / "train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(root / "val",   transform=val_tfms)

    CLASSES = train_ds.classes
    print("ImageFolder classes (order matters):", CLASSES)

    assert val_ds.classes == CLASSES, "Train/val class orders must match!"

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4)

    num_classes = len(CLASSES)
    return train_loader, val_loader, num_classes


def create_model(num_classes: int):
    print(f"Creating model: {ARCH} with num_classes={num_classes}")
    model = timm.create_model(
        ARCH,
        pretrained=True,
        num_classes=num_classes,
    )
    return model


def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    print("Loading dataset from:", DATA_ROOT)
    train_loader, val_loader, num_classes = get_dataloaders(DATA_ROOT)
    model = create_model(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        dt = time.time() - t0

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
            f"time={dt:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> New best model saved to {CHECKPOINT_PATH} (val_acc={val_acc:.3f})")

    print("Training complete. Best val_acc:", best_val_acc)
    print("Final classes order:", CLASSES)


if __name__ == "__main__":
    main()
