# 🚗 Lane Detection — YOLO11x-seg on IDD Dataset

> Real-time road and lane segmentation for Indian driving conditions using state-of-the-art instance segmentation.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-green)](https://ultralytics.com)
[![Dataset](https://img.shields.io/badge/Dataset-IDD%2020k%20Part%20II-yellow)](https://idd.insaan.iiit.ac.in/)

---

## 📌 Overview

This project trains a **YOLO11x-seg** (X-Large segmentation) model to detect and segment road surfaces from dashcam video footage. It is specifically designed and evaluated on the **India Driving Dataset (IDD 20k Part II)**, which captures the complex and unstructured nature of Indian roads.

The trained model can be used for:
- Autonomous vehicle perception pipelines
- Driver assistance systems (ADAS)
- Road condition monitoring

---

## 🏗️ Project Structure

```
Lane-Detection/
├── lane_detection.ipynb     # Main training notebook
├── idd_lane.yaml            # YOLO dataset configuration
├── .gitignore               # Excludes data and caches
├── .gitattributes           # Git LFS configuration for *.pt files
├── yolo11x-seg.pt           # Pre-trained YOLO11x-seg base model (via LFS)
├── yolo26n.pt               # AMP-check utility model (via LFS)
├── runs/                    # Training outputs (ignored by git)
│   └── segment/
│       └── lane_project/
│           └── v1_highres/
│               ├── weights/
│               │   ├── best.pt   # Best checkpoint
│               │   └── last.pt   # Latest checkpoint
│               └── results.csv
└── Data/                    # Dataset (ignored by git — download separately)
    └── idd20kII/
        ├── leftImg8bit/     # Raw dashcam images
        └── gtFine/          # Ground truth polygon annotations
```

---

## 📦 Dataset

**India Driving Dataset (IDD) — 20k Part II**
- **Source:** [IDD Official Website](https://idd.insaan.iiit.ac.in/)
- **Train split:** 6,830 images
- **Val split:** 1,022 images
- **Image format:** JPEG (1920×1080)
- **Annotation format:** JSON polygon annotations (`_gtFine_polygons.json`)
- **Target class:** `road`

> ⚠️ The dataset is not included in this repository due to size. Download it separately and place it at `Data/idd20kII/`.

---

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/agusrajuthaliyan/Lane-Detection.git
cd Lane-Detection
```

### 2. Install Dependencies
```bash
pip install ultralytics opencv-python tqdm torch torchvision
```

### 3. Download and Place the Dataset
Download **IDD 20k Part II** from the [official website](https://idd.insaan.iiit.ac.in/) and extract it to:
```
Lane-Detection/Data/idd20kII/
```

---

## 🚀 Training

### Step 1 — Generate YOLO Labels (run once)
Open `lane_detection.ipynb` and run **Cell 2**. This converts the IDD JSON polygon annotations into YOLO segmentation format (`.txt` files placed next to each image).

```
✅ Generated 6830 YOLO labels for train.
✅ Generated 1022 YOLO labels for val.
```

### Step 2 — Start Training (run Cell 3)
```python
from ultralytics import YOLO
import time, datetime

model = YOLO('yolo11x-seg.pt')
start_time = time.time()

results = model.train(
    data='idd_lane.yaml',
    epochs=100,
    imgsz=1024,
    batch=8,
    device=0,
    workers=16,
    project='lane_project',
    name='v1_highres',
    exist_ok=True,
    amp=True
)

total = str(datetime.timedelta(seconds=int(time.time() - start_time)))
print(f"✅ Training complete! Time taken: {total}")
print(f"Best weights: {results.save_dir}/weights/best.pt")
```

**Expected training time:** ~30–32 hours on an RTX 4000 SFF Ada at 1.6 it/s.

---

## ⏸️ Pausing and Resuming Training

### To Pause:
1. Stop the training cell (click ⏹ in Jupyter)
2. Save and push the latest checkpoint:
```bash
git add runs/segment/lane_project/v1_highres/weights/last.pt
git commit -m "Checkpoint: pausing at epoch X"
git push origin main
```

### To Resume:
```python
model = YOLO('runs/segment/lane_project/v1_highres/weights/last.pt')
results = model.train(resume=True)
```

---

## 🧠 Model

| Property | Value |
|---|---|
| Architecture | YOLO11x-seg |
| Parameters | 62,051,411 |
| GFLOPs | 297.0 |
| Input Size | 1024×1024 |
| Task | Instance Segmentation |
| Optimizer | AdamW (auto) |
| Precision | AMP (Mixed Precision) |
| Pre-trained On | COCO |
| Fine-tuned On | IDD 20k Part II |

---

## 🖥️ Hardware Used

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4000 SFF Ada Generation |
| VRAM | 20 GB GDDR6 |
| CPU | Intel i7-13700KF (24 threads) |
| OS | Ubuntu Linux |
| CUDA | 13.0 |
| Python | 3.10.12 |

---

## 📤 Pushing Updates to GitHub

```bash
git add .
git commit -m "describe your changes"
git push origin main
```

> Large `.pt` files are handled automatically via **Git LFS**.

---

## 📄 License

This project is for research and educational purposes. The IDD dataset is subject to its own [usage terms](https://idd.insaan.iiit.ac.in/).
