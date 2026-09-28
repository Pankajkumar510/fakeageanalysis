# FakeAgeAnalysis

FakeAgeAnalysis is a FastAPI-based video analysis prototype with three machine-learning components:

- **Age estimation:** ResNet18 regression over sampled video frames.
- **Deepfake detection:** ResNet18 binary classification over sampled video frames.
- **rPPG heart-rate estimation:** a 1D convolutional model over a normalized green-channel signal.

The project analyzes an uploaded video for suspected fake content, estimates age, and provides a non-diagnostic health analysis from estimated heart rate.

The browser interface is in `frontend/index.html`. The API is in `api.py`; Streamlit is no longer used.

## Current Results

These are the recorded values in `models/evaluation_metrics.json`:

| Model | Evaluation | Result |
|---|---|---:|
| Age | 1,526 validation samples, MAE | 5.31 years |
| Age | Within 5 years | 14.98% |
| Deepfake | 2,000 test images, accuracy | 89.05% |
| Deepfake | Precision / recall / F1 | 91.59% / 86.00% / 88.71% |
| rPPG | 4 validation samples, MAE | 74.66 BPM |

The rPPG metric is not production meaningful because it is based on only four validation samples. See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the detailed assessment, dataset inventory, model-efficiency discussion, risks, and production roadmap.

## Dataset

Expected local layout:

- `Data/UTKface/` and `Data/train.csv`, `Data/val.csv` for age estimation.
- `Data/DeepFake/` and the deepfake train/validation/test CSV files for deepfake detection.
- `Data/rPPZ/` for paired video and Empatica BVP recordings.

The current inventory contains 6,106 age training rows, 1,526 age validation rows, 16,000 deepfake training images, 2,000 deepfake validation images, 2,000 deepfake test images, and 10 rPPG videos. Raw datasets are excluded from Git by `.gitignore`.

## Installation

```powershell
py -3 -m pip install -r requirements.txt
```

For model evaluation, install the optional evaluation dependency as well:

```powershell
py -3 -m pip install -r requirements-dev.txt
```

## Run The API

```powershell
py -3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` after the server starts. The API exposes:

- `GET /api/status`
- `POST /api/analyze` with a video upload in the `file` field

## Train

Pretrained ResNet18 initialization is enabled by default. The age pipeline excludes labels outside 0-100, and the rPPG pipeline supports both `empatica_e4` and `empatica_data` sensor directories.

```powershell
py -3 train_age_deepfake.py --model both --epochs 20
py -3 train_rppg.py
```

Use `--no-pretrained` only when intentionally training from random initialization.

## Evaluate

```powershell
py -3 evaluate_models.py
```

This writes updated metrics to `models/evaluation_metrics.json`. Always record the dataset version, checkpoint, configuration, and split used for each result.

## Deployment Notes

The recommended deployment is Render as one Docker web service. `Dockerfile` installs the CPU PyTorch/OpenCV runtime, copies the models, starts FastAPI, and serves the static frontend from the same origin. `render.yaml` provides the Render blueprint and uses `/api/status` as the health check.

To deploy:

1. Open https://dashboard.render.com/select-repo?type=blueprint.
2. Connect `Pankajkumar510/fakeageanalysis`.
3. Select `render.yaml` and create the service.
4. Wait for the Docker build, then open the generated Render URL.

The free Render tier may sleep when idle and CPU inference may be slow. Use a paid instance for reliable availability and higher upload/inference workloads. `vercel.json` remains available for a frontend-only deployment, but Vercel cannot run this PyTorch/OpenCV backend as a normal serverless function.

## Production Status

This is a research/prototype system, not a clinical or production decision system. The largest blockers are the small and weakly validated rPPG dataset, frame-level rather than video-level modeling, lack of face detection and tracking, limited robustness testing, permissive API security settings, and missing latency/load monitoring. The full recommendations are in [PROJECT_REPORT.md](PROJECT_REPORT.md).
