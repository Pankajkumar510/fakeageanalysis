import os

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

st.set_page_config(page_title="Video Analysis", page_icon="🎥", layout="wide")

st.markdown(
    """
    <style>
    .main { background: linear-gradient(180deg, #0b1020 0%, #111827 100%); }
    .block-container { padding-top: 2rem; }
    .hero {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0 0 0.3rem; color: #f8fafc; }
    .hero p { margin: 0; color: #cbd5e1; }
    .card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 1rem;
        height: 100%;
    }
    .label { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .value { color: #f8fafc; font-size: 1.5rem; font-weight: 700; }
    .success { background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(74, 222, 128, 0.4); border-radius: 12px; padding: 1rem; }
    .warning { background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(251, 191, 36, 0.4); border-radius: 12px; padding: 1rem; }
    .error { background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(248, 113, 113, 0.4); border-radius: 12px; padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_age_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    path = "models/age_model.pth"
    if not os.path.exists(path):
        return None
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_resource
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


@st.cache_resource
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


st.markdown(
    """
    <div class="hero">
        <h1>Video Health and Authenticity Check</h1>
        <p>Upload a short video to get an overall assessment for age, deepfake authenticity, and heart-health.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_cols = st.columns(3)
with status_cols[0]:
    st.markdown(
        """
        <div class="card">
            <div class="label">Age Model</div>
            <div class="value">%s</div>
        </div>
        """ % ("Ready" if age_model is not None else "Missing"),
        unsafe_allow_html=True,
    )
with status_cols[1]:
    st.markdown(
        """
        <div class="card">
            <div class="label">Deepfake Model</div>
            <div class="value">%s</div>
        </div>
        """ % ("Ready" if deepfake_model is not None else "Missing"),
        unsafe_allow_html=True,
    )
with status_cols[2]:
    st.markdown(
        """
        <div class="card">
            <div class="label">Heart Model</div>
            <div class="value">%s</div>
        </div>
        """ % ("Ready" if rppg_model is not None else "Missing"),
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("Upload")
    video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "m4v"], accept_multiple_files=False)
    run = st.button("Analyze Video", type="primary", use_container_width=True)

if video is not None:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, video.name)
    with open(temp_path, "wb") as f:
        f.write(video.read())

    st.video(temp_path)

    if run:
        with st.spinner("Running analysis..."):
            age = estimate_age_from_video(temp_path)
            deepfake_label, deepfake_score = estimate_deepfake_from_video(temp_path)
            heart_rate = estimate_heart_rate(temp_path)
            health, message = health_status(heart_rate)

        col1, col2, col3 = st.columns(3)
        with col1:
            if age is not None:
                st.markdown(f"<div class='success'><strong>Estimated Age:</strong> {age:.1f} years</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='error'>Age could not be estimated.</div>", unsafe_allow_html=True)

        with col2:
            if deepfake_label is not None:
                confidence = deepfake_score * 100 if deepfake_score is not None else 0
                if deepfake_label == "Real":
                    st.markdown(f"<div class='success'><strong>Authenticity:</strong> {deepfake_label}<br><strong>Confidence:</strong> {confidence:.1f}% real</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='error'><strong>Authenticity:</strong> {deepfake_label}<br><strong>Confidence:</strong> {confidence:.1f}% fake</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='error'>Deepfake check could not run.</div>", unsafe_allow_html=True)

        with col3:
            if health == "Healthy":
                st.markdown(f"<div class='success'><strong>{health}</strong><br>{message}</div>", unsafe_allow_html=True)
            elif health == "Borderline":
                st.markdown(f"<div class='warning'><strong>{health}</strong><br>{message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='error'><strong>{health}</strong><br>{message}</div>", unsafe_allow_html=True)

        st.markdown("### Final Result")
        summary = [
            {"Metric": "Age", "Value": f"{age:.1f} years" if age is not None else "N/A"},
            {"Metric": "Deepfake Status", "Value": deepfake_label if deepfake_label is not None else "N/A"},
            {"Metric": "Heart Rate", "Value": f"{heart_rate:.1f} bpm" if heart_rate is not None else "N/A"},
            {"Metric": "Health Status", "Value": health},
        ]
        st.dataframe(summary, use_container_width=True, hide_index=True)
else:
    st.info("Upload a video to start the analysis.")
