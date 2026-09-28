import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil

app = FastAPI(title="Video Analysis API")

# Enable CORS for the Vue frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_age_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    path = "models/age_model.pth"
    if not os.path.exists(path):
        return None
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model

def load_deepfake_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    path = "models/deepfake_model.pth"
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_rppg_model():
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

    model = PulseRegressor()
    path = "models/rppg_model.pth"
    if not os.path.exists(path):
        return None
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model

# Global models
age_model = load_age_model()
deepfake_model = load_deepfake_model()
rppg_model = load_rppg_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def extract_green_signal(video_path, max_frames=300):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    signal = []
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_count >= max_frames:
            break
        frame_count += 1
        if frame_count % 2 != 0:
            continue
        if frame is None:
            continue

        h, w = frame.shape[:2]
        if h < 40 or w < 40:
            continue

        top, bottom = h // 6, h - h // 6
        left, right = w // 6, w - w // 6
        roi = frame[top:bottom, left:right]
        if roi.size == 0:
            continue

        green = roi[:, :, 1].astype(np.float32)
        signal.append(float(np.mean(green)))

    capture.release()
    if len(signal) < 32:
        return None

    arr = np.asarray(signal, dtype=np.float32)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return arr if std < 1e-6 else (arr - mean) / std

def estimate_age_from_video(video_path):
    if age_model is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    ages = []
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_count % 6 != 0:
            frame_count += 1
            continue

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            ages.append(age_model(tensor).item())

    cap.release()
    return None if not ages else float(np.mean(ages))

def estimate_deepfake_from_video(video_path):
    if deepfake_model is None:
        return None, None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    fake_scores = []
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_count % 8 != 0:
            frame_count += 1
            continue

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = deepfake_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            fake_scores.append(float(probs[1].item()))

    cap.release()
    if not fake_scores:
        return None, None

    avg = float(np.mean(fake_scores))
    label = "Fake" if avg >= 0.5 else "Real"
    return label, avg

def estimate_heart_rate(video_path):
    if rppg_model is None:
        return None

    signal = extract_green_signal(video_path)
    if signal is None:
        return None

    signal = signal[-64:]
    if signal.shape[0] < 64:
        pad = np.zeros(64, dtype=np.float32)
        pad[: signal.shape[0]] = signal
        signal = pad

    tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return float(rppg_model(tensor).item())

def health_status(hr):
    if hr is None:
        return "Unavailable", "Heart rate could not be estimated from the uploaded video."
    if 60 <= hr <= 100:
        return "Healthy", f"Estimated heart rate: {hr:.1f} bpm"
    if 50 <= hr < 60 or 100 < hr <= 110:
        return "Borderline", f"Estimated heart rate: {hr:.1f} bpm — slightly outside the normal range."
    return "Needs Attention", f"Estimated heart rate: {hr:.1f} bpm — outside the normal range."


@app.get("/api/status")
def get_status():
    return {
        "age_model": "Ready" if age_model is not None else "Missing",
        "deepfake_model": "Ready" if deepfake_model is not None else "Missing",
        "heart_model": "Ready" if rppg_model is not None else "Missing",
    }

@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        age = estimate_age_from_video(temp_path)
        deepfake_label, deepfake_score = estimate_deepfake_from_video(temp_path)
        heart_rate = estimate_heart_rate(temp_path)
        health, message = health_status(heart_rate)
        
        return {
            "success": True,
            "results": {
                "age": round(age, 1) if age is not None else None,
                "deepfake_label": deepfake_label,
                "deepfake_score": round(deepfake_score, 4) if deepfake_score is not None else None,
                "heart_rate": round(heart_rate, 1) if heart_rate is not None else None,
                "health_status": health,
                "health_message": message
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Serve the frontend at the root (must be after API routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
