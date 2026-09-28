import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def resolve_dataset_path(value):
    relative = Path(value.replace("\\", "/"))
    candidates = [ROOT / relative, ROOT / "Data" / relative.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_age_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(ROOT / "models/age_model.pth", map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def load_deepfake_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(ROOT / "models/deepfake_model.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    return model.to(DEVICE).eval()


class PulseRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def load_rppg_model():
    model = PulseRegressor()
    model.load_state_dict(torch.load(ROOT / "models/rppg_model.pth", map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def evaluate_age():
    model = load_age_model()
    targets = []
    predictions = []
    tensors = []
    with (ROOT / "Data/val.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            image_path = resolve_dataset_path(row["image_path"])
            if image_path is None:
                continue
            image = Image.open(image_path).convert("RGB")
            tensors.append(IMAGE_TRANSFORM(image))
            targets.append(float(row["age"]))
    with torch.no_grad():
        for start in range(0, len(tensors), 64):
            batch = torch.stack(tensors[start : start + 64]).to(DEVICE)
            predictions.extend(model(batch).cpu().tolist())
    return {
        "samples": len(targets),
        "mae_years": float(mean_absolute_error(targets, predictions)),
        "within_5_years_percent": float(np.mean(np.abs(np.asarray(targets) - predictions) <= 5) * 100),
    }


def evaluate_deepfake():
    model = load_deepfake_model()
    targets = []
    predictions = []
    tensors = []
    with (ROOT / "Data/deepfake_test.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            image_path = resolve_dataset_path(row["image_path"])
            if image_path is None:
                continue
            image = Image.open(image_path).convert("RGB")
            tensors.append(IMAGE_TRANSFORM(image))
            targets.append(int(row["label"]))
    with torch.no_grad():
        for start in range(0, len(tensors), 64):
            batch = torch.stack(tensors[start : start + 64]).to(DEVICE)
            predictions.extend(model(batch).argmax(dim=1).cpu().tolist())
    return {
        "samples": len(targets),
        "accuracy_percent": float(accuracy_score(targets, predictions) * 100),
        "precision_percent": float(precision_score(targets, predictions, zero_division=0) * 100),
        "recall_percent": float(recall_score(targets, predictions, zero_division=0) * 100),
        "f1_percent": float(f1_score(targets, predictions, zero_division=0) * 100),
    }


def load_bvp_signal(csv_path):
    with csv_path.open(encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    sample_rate = float(lines[1]) if len(lines) > 1 else 64.0
    values = []
    for line in lines[2:]:
        for value in line.split(","):
            try:
                values.append(float(value.strip()))
            except ValueError:
                continue
    return np.asarray(values, dtype=np.float32), sample_rate


def compute_heart_rate(signal, sample_rate):
    signal = signal - np.mean(signal)
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal))
    mask = (frequencies >= 0.7) & (frequencies <= 3.0)
    return float(frequencies[mask][np.argmax(spectrum[mask])] * 60.0)


def extract_green_signal(video_path):
    capture = cv2.VideoCapture(str(video_path))
    signal = []
    frame_count = 0
    while True:
        success, frame = capture.read()
        if not success or frame_count >= 300:
            break
        frame_count += 1
        if frame_count % 2 != 0:
            continue
        height, width = frame.shape[:2]
        roi = frame[height // 6 : height - height // 6, width // 6 : width - width // 6]
        if roi.size:
            signal.append(float(np.mean(roi[:, :, 1])))
    capture.release()
    if len(signal) < 32:
        return None
    signal = np.asarray(signal, dtype=np.float32)
    standard_deviation = float(np.std(signal))
    return signal if standard_deviation < 1e-6 else (signal - np.mean(signal)) / standard_deviation


def evaluate_rppg():
    model = load_rppg_model()
    records = []
    video_files = sorted((ROOT / "Data/rPPZ").glob("**/video/*.MOV"))
    video_files += sorted((ROOT / "Data/rPPZ").glob("**/video/*.mov"))
    video_files += sorted((ROOT / "Data/rPPZ").glob("**/video/*.mp4"))
    for video_path in video_files:
        bvp_path = video_path.parent.parent / "empatica_e4" / "BVP.csv"
        signal = extract_green_signal(video_path)
        if not bvp_path.exists() or signal is None:
            continue
        bvp, sample_rate = load_bvp_signal(bvp_path)
        if len(bvp) < 64:
            continue
        signal = signal[-64:]
        if len(signal) < 64:
            padded = np.zeros(64, dtype=np.float32)
            padded[: len(signal)] = signal
            signal = padded
        target = compute_heart_rate(bvp[-64:], sample_rate)
        if 40 <= target <= 180:
            records.append((signal, target))

    split = max(1, int(len(records) * 0.8))
    validation_records = records[split:]
    targets = []
    predictions = []
    with torch.no_grad():
        for signal, target in validation_records:
            tensor = torch.tensor(signal).float().view(1, 1, -1).to(DEVICE)
            predictions.append(float(model(tensor).item()))
            targets.append(target)
    return {
        "samples": len(targets),
        "mae_bpm": float(mean_absolute_error(targets, predictions)) if targets else None,
    }


def main():
    metrics = {
        "age": evaluate_age(),
        "deepfake": evaluate_deepfake(),
        "rppg": evaluate_rppg(),
    }
    output_path = ROOT / "models/evaluation_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()