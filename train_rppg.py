import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = Path("Data/rPPZ")
MODEL_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-3
WINDOW_LENGTH = 64
STEP_LENGTH = 32


class RPPGDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        signal, target = self.records[idx]
        signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(float(target), dtype=torch.float32)
        return signal, target


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
        x = self.net(x)
        return self.head(x).squeeze(-1)


def load_bvp_signal(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if len(lines) < 3:
        raise ValueError(f"BVP file is too short: {csv_path}")

    try:
        sample_rate = float(lines[1])
    except ValueError:
        sample_rate = 64.0

    values = []
    for line in lines[2:]:
        if "," in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            values.extend(float(p) for p in parts)
        else:
            try:
                values.append(float(line))
            except ValueError:
                continue

    if not values:
        raise ValueError(f"No numeric BVP values found in {csv_path}")
    return np.asarray(values, dtype=np.float32), float(sample_rate)


def compute_heart_rate(signal, sample_rate):
    signal = signal - np.mean(signal)
    if len(signal) < 16:
        return 0.0

    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal))
    mask = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(mask):
        return 0.0

    dominant = freqs[mask][np.argmax(spectrum[mask])]
    return float(dominant * 60.0)


def normalize_signal(signal):
    if signal.size == 0:
        return signal
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    if std < 1e-6:
        return signal
    return (signal - mean) / std


def extract_face_signal(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    signal = []
    frame_count = 0
    max_frames = 300

    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_count >= max_frames:
            break
        frame_count += 1

        if frame_count % 2 != 0:
            continue

        if frame is None:
            continue

        height, width = frame.shape[:2]
        if height < 40 or width < 40:
            continue

        top = height // 6
        bottom = height - height // 6
        left = width // 6
        right = width - width // 6
        roi = frame[top:bottom, left:right]
        if roi.size == 0:
            continue

        green = roi[:, :, 1].astype(np.float32)
        signal.append(float(np.mean(green)))

    capture.release()

    if len(signal) < 32:
        return None

    signal = np.asarray(signal, dtype=np.float32)
    signal = normalize_signal(signal)
    return signal


def build_dataset():
    records = []
    video_files = sorted(DATA_DIR.glob("**/video/*.MOV")) + sorted(DATA_DIR.glob("**/video/*.mov")) + sorted(DATA_DIR.glob("**/video/*.mp4"))

    for video_path in video_files:
        bvp_path = video_path.parent.parent / "empatica_e4" / "BVP.csv"
        if not bvp_path.exists():
            continue

        green_signal = extract_face_signal(video_path)
        if green_signal is None:
            continue

        bvp_signal, sample_rate = load_bvp_signal(bvp_path)
        if len(bvp_signal) < WINDOW_LENGTH:
            continue

        max_start = max(0, len(green_signal) - WINDOW_LENGTH)
        for start in range(0, max_start + 1, STEP_LENGTH):
            signal_window = green_signal[start : start + WINDOW_LENGTH]
            if len(signal_window) < WINDOW_LENGTH:
                continue

            # Align the BVP-derived heart-rate estimate to the same time span as the sampled video window.
            signal_ratio = len(green_signal) / max(len(bvp_signal), 1)
            bvp_start = int(start / max(len(green_signal), 1) * len(bvp_signal))
            bvp_end = min(len(bvp_signal), bvp_start + WINDOW_LENGTH)
            if bvp_end - bvp_start < 16:
                continue
            bvp_segment = bvp_signal[bvp_start:bvp_end]
            target_hr = compute_heart_rate(bvp_segment, sample_rate)

            if 40 <= target_hr <= 180:
                records.append((signal_window.astype(np.float32), float(target_hr)))

    if not records:
        raise RuntimeError(f"No usable rPPG training samples found under {DATA_DIR}")

    return records


def main():
    records = build_dataset()
    split_index = max(1, int(len(records) * 0.8))
    train_records = records[:split_index]
    val_records = records[split_index:]

    train_loader = DataLoader(RPPGDataset(train_records), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RPPGDataset(val_records), batch_size=BATCH_SIZE, shuffle=False)

    model = PulseRegressor().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_mae = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for signals, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            signals = signals.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(signals)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * signals.size(0)

        train_loss /= len(train_records)

        model.eval()
        total_error = 0.0
        total_count = 0
        with torch.no_grad():
            for signals, targets in val_loader:
                signals = signals.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = model(signals)
                total_error += torch.abs(predictions - targets).sum().item()
                total_count += signals.size(0)

        mae = total_error / max(total_count, 1)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_mae={mae:.2f} bpm")

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), MODEL_DIR / "rppg_model.pth")
            print(f"Saved best model to {MODEL_DIR / 'rppg_model.pth'}")

    print(f"Training complete. Best validation MAE: {best_mae:.2f} bpm")


if __name__ == "__main__":
    main()
