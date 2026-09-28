import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 64 if DEVICE.type == "cuda" else 32


class CsvImageDataset(Dataset):
    def __init__(self, rows, transform, target_name):
        self.rows = rows
        self.transform = transform
        self.target_name = target_name

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(resolve_path(row["image_path"])).convert("RGB")
        return self.transform(image), float(row[self.target_name])


def resolve_path(value):
    relative = Path(value.replace("\\", "/"))
    candidates = [ROOT / relative, ROOT / "Data" / relative.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def read_rows(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(training):
    if training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.7),
            transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_resnet(output_count, pretrained):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = models.resnet18(weights=weights)
    except Exception as error:
        print(f"Pretrained weights unavailable, using random initialization: {error}")
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, output_count)
    return model.to(DEVICE)


def evaluate_age(model, loader):
    model.eval()
    absolute_error = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, targets in loader:
            predictions = model(images.to(DEVICE)).squeeze(1)
            absolute_error += torch.abs(predictions - targets.to(DEVICE)).sum().item()
            sample_count += targets.size(0)
    return absolute_error / max(sample_count, 1)


def train_age(epochs, pretrained):
    train_rows = read_rows(ROOT / "Data/train.csv")
    validation_rows = read_rows(ROOT / "Data/val.csv")
    train_dataset = CsvImageDataset(train_rows, build_transforms(True), "age")
    validation_dataset = CsvImageDataset(validation_rows, build_transforms(False), "age")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_resnet(1, pretrained)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.SmoothL1Loss(beta=2.0)
    best_mae = float("inf")
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images.to(DEVICE)).squeeze(1)
            loss = criterion(predictions, targets.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running_loss += loss.item() * targets.size(0)

        validation_mae = evaluate_age(model, validation_loader)
        scheduler.step(validation_mae)
        train_loss = running_loss / len(train_dataset)
        print(f"Age epoch {epoch + 1}/{epochs}: loss={train_loss:.4f}, val_mae={validation_mae:.3f}")
        if validation_mae < best_mae:
            best_mae = validation_mae
            patience_count = 0
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / "age_model.pth")
        else:
            patience_count += 1
            if patience_count >= 5:
                break
    print(f"Age training complete. Best validation MAE: {best_mae:.3f} years")


def evaluate_deepfake(model, loader):
    model.eval()
    correct_count = 0
    sample_count = 0
    with torch.no_grad():
        for images, targets in loader:
            predictions = model(images.to(DEVICE)).argmax(dim=1).cpu()
            correct_count += (predictions == targets.long()).sum().item()
            sample_count += targets.size(0)
    return correct_count / max(sample_count, 1)


def train_deepfake(epochs, pretrained):
    train_rows = read_rows(ROOT / "Data/deepfake_train.csv")
    validation_rows = read_rows(ROOT / "Data/deepfake_validation.csv")
    train_dataset = CsvImageDataset(train_rows, build_transforms(True), "label")
    validation_dataset = CsvImageDataset(validation_rows, build_transforms(False), "label")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_counts = np.bincount([int(row["label"]) for row in train_rows], minlength=2)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float32).to(DEVICE)
    model = build_resnet(2, pretrained)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    best_accuracy = 0.0
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(DEVICE))
            loss = criterion(logits, targets.long().to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running_loss += loss.item() * targets.size(0)

        validation_accuracy = evaluate_deepfake(model, validation_loader)
        scheduler.step(validation_accuracy)
        train_loss = running_loss / len(train_dataset)
        print(f"Deepfake epoch {epoch + 1}/{epochs}: loss={train_loss:.4f}, val_accuracy={validation_accuracy * 100:.2f}%")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            patience_count = 0
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, MODEL_DIR / "deepfake_model.pth")
        else:
            patience_count += 1
            if patience_count >= 5:
                break
    print(f"Deepfake training complete. Best validation accuracy: {best_accuracy * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["age", "deepfake", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    seed_everything(arguments.seed)
    if arguments.model in ["age", "both"]:
        train_age(arguments.epochs, arguments.pretrained)
    if arguments.model in ["deepfake", "both"]:
        train_deepfake(arguments.epochs, arguments.pretrained)


if __name__ == "__main__":
    main()
