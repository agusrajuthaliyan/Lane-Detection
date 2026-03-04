# Lane Detection Project Report

**Project:** Road and Lane Segmentation using YOLO11x-seg  
**Author:** Agus Rajuth Aliyan  
**GitHub:** [agusrajuthaliyan/Lane-Detection](https://github.com/agusrajuthaliyan/Lane-Detection)  
**Date:** March 2026  

---

## 1. Project Summary

This project develops a deep learning-based **road and lane segmentation** model tailored to Indian driving conditions. It uses the **YOLO11x-seg** (X-Large instance segmentation) architecture, fine-tuned on the **India Driving Dataset (IDD) 20k Part II**, to detect and segment drivable road surfaces in real time from dashcam footage.

The primary goal is to explore and demonstrate how a state-of-the-art YOLO model can be adapted to the complex, unstructured, and often chaotic road environments common in India — including mixed traffic, unmarked lanes, and variable road quality.

---

## 2. Problem Statement

Lane detection in India presents unique challenges compared to Western driving environments:

- **Absence of lane markings** on a large proportion of roads
- **Mixed traffic** including pedestrians, animals, and non-motorized vehicles
- **Highly variable road quality** — potholes, unpaved sections, and dust
- **Unstructured driving behaviour** — vehicles driving on shoulders or wrong sides

Traditional lane detection pipelines, built for structured Western roads (e.g., IPM + Hough transforms), fail in these conditions. A learning-based segmentation approach is essential.

---

## 3. Dataset

### 3.1 India Driving Dataset (IDD) — 20k Part II

| Property | Details |
|---|---|
| **Source** | IIIT Hyderabad — [idd.insaan.iiit.ac.in](https://idd.insaan.iiit.ac.in/) |
| **Total images** | ~20,000 (Part II subset used) |
| **Training images** | 6,830 |
| **Validation images** | 1,022 |
| **Image resolution** | 1920 × 1080 pixels |
| **Image format** | JPEG |
| **Annotation format** | JSON polygon files (`_gtFine_polygons.json`) |
| **Scene coverage** | Urban, rural, highway, night, rain |

### 3.2 Label Classes Used

The IDD dataset contains 40+ semantic classes. For this project, only the **`road`** class is extracted (the paved drivable surface ahead), converting a multi-class segmentation problem into a focused binary segmentation task.

### 3.3 Label Conversion Pipeline

The raw IDD annotations provide polygon coordinates in JSON format. A custom conversion script was written (Cell 2 of the notebook) to:

1. Parse each `_gtFine_polygons.json` file
2. Extract all polygon objects with label `"road"`
3. Normalize polygon coordinates to the range `[0.0, 1.0]`
4. Write YOLO-format segmentation labels (`.txt`) directly alongside each image

```
Output format per line:
0 x1 y1 x2 y2 ... xN yN   (class_id followed by normalized polygon points)
```

This produced:
- **6,830 `.txt` label files** in the training split
- **1,022 `.txt` label files** in the validation split

---

## 4. Model Architecture

### 4.1 YOLO11x-seg (X-Large Segmentation)

YOLO11 is Ultralytics' latest generation YOLO model, released in 2024. The `x` variant is the largest model in the family, offering the highest accuracy at the cost of compute.

| Property | Value |
|---|---|
| **Model** | YOLO11x-seg |
| **Total parameters** | 62,051,411 |
| **GFLOPs** | 297.0 |
| **Backbone** | C3k2 + C2PSA (improved CSP with attention) |
| **Neck** | PAN-FPN (multi-scale feature fusion) |
| **Head** | Segment head (detection + mask prototype) |
| **Mask prototypes** | 32 |
| **Output** | Bounding boxes + instance segmentation masks |

### 4.2 Transfer Learning

The model was initialized with **COCO pre-trained weights** (`yolo11x-seg.pt`). Only the final segmentation head was re-initialized for `nc=1` (one class: road). All 1,071 out of 1,077 weight tensors were successfully transferred from the pre-trained model.

---

## 5. Training Configuration

### 5.1 Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| `epochs` | 100 | Standard for convergence on this dataset size |
| `imgsz` | 1024 | High resolution to preserve thin lane boundaries |
| `batch` | 8 | Auto-reduced from 16 due to VRAM constraints at 1024px |
| `optimizer` | AdamW (auto) | lr=0.002, momentum=0.9 |
| `amp` | True | Mixed precision for RTX Ada Tensor Cores |
| `workers` | 16 | Matches CPU thread count for fast data loading |
| `patience` | 100 | No early stopping (full training run) |
| `augment` | True | Mosaic, fliplr, HSV augmentations |

### 5.2 Hardware

| Component | Specification |
|---|---|
| **GPU** | NVIDIA RTX 4000 SFF Ada Generation |
| **VRAM** | 20 GB GDDR6 |
| **GPU CUDA Cores** | 6,144 |
| **Tensor Cores** | 4th generation (Ada Lovelace) |
| **CPU** | Intel i7-13700KF |
| **CPU Threads** | 24 |
| **OS** | Ubuntu Linux |
| **CUDA Version** | 13.0 |
| **PyTorch Version** | 2.10.0+cu130 |
| **Python** | 3.10.12 |

### 5.3 Training Performance

| Metric | Value |
|---|---|
| **Iterations per second** | ~1.6 it/s |
| **Batches per epoch** | ~1,759 |
| **Time per epoch** | ~18–19 minutes |
| **Total estimated time** | ~30–32 hours |
| **GPU memory used** | ~11.8 GB / 20 GB |

---

## 6. Challenges Encountered and Solutions

### 6.1 YAML Configuration Error
**Problem:** The `idd_lane.yaml` file was accidentally written as a Python code file (f-string assignment) instead of valid YAML. YOLO's dataset loader raised `ScannerError: mapping values are not allowed in this context`.  
**Solution:** The file was rewritten as a clean valid YAML configuration.

### 6.2 Missing Ground Truth Directory (`gtFine`)
**Problem:** The initial dataset download only contained the images (`leftImg8bit`). The label annotation folder (`gtFine`) was a separate download and was absent.  
**Solution:** The Part II download was obtained, which contained the full `gtFine` JSON polygon annotations alongside the corresponding images in a single archive.

### 6.3 Empty Label Files
**Problem:** The original label generation script wrote labels to a separate `idd_lane_labels/` folder using `_labelids.png` mask images (looking for a specific pixel value). Without `gtFine`, no labels were generated, causing the "No labels found" warning.  
**Solution:** A new JSON-based label conversion script was written that directly reads the polygon coordinates from `_gtFine_polygons.json` files and writes YOLO `.txt` labels next to the images.

### 6.4 CUDA Library Mismatch
**Problem:** PyTorch was compiled against CUDA 13.0, but the installed `libnvrtc-builtins.so` was from CUDA 12.8 (under `nvidia/cuda_nvrtc/`). The correct `libnvrtc-builtins.so.13.0` existed but was not on `LD_LIBRARY_PATH`.  
**Solution:** Pre-load the correct library at kernel startup using `ctypes.CDLL()` before any torch/ultralytics import, bypassing the dynamic linker search path issue.

### 6.5 Git Attribution on Shared Machine
**Problem:** The shared coworking workstation had another user's Git identity configured globally, causing the initial commit to be attributed to `Mark-Joseph-42`.  
**Solution:** Used `git config --local` to set the correct author identity scoped to this repository only, then amended and force-pushed the commit.

---

## 7. Repository Management

### 7.1 Git LFS for Large Files
Model weight files (`*.pt`) are tracked using **Git Large File Storage (LFS)** to stay within GitHub's 100 MB per-file limit. LFS stores the binary content on GitHub's LFS servers while the repository contains lightweight pointer files.

### 7.2 .gitignore Strategy
| Excluded | Reason |
|---|---|
| `/Data` | ~6 GB dataset — too large, user must download separately |
| `/runs` | Training artifacts — large, regenerable |
| `*.cache` | YOLO dataset cache files |
| `mlruns/` | MLflow experiment tracking logs |

---

## 8. Pause-and-Resume Workflow

Since training is conducted on a shared workstation at a coworking space, training is split across multiple sessions:

### Pause
```bash
git add runs/segment/lane_project/v1_highres/weights/last.pt
git commit -m "Checkpoint: epoch X"
git push origin main
```

### Resume
```python
model = YOLO('runs/segment/lane_project/v1_highres/weights/last.pt')
results = model.train(resume=True)
```

YOLO's built-in checkpoint resumption restores the exact epoch number, optimizer state, and scheduler state automatically.

---

## 9. Next Steps

1. **Complete 100-epoch training run** and evaluate on the validation set
2. **Export to TensorRT** (`.engine`) for optimized real-time inference on the RTX 4000
3. **Run inference on video** — process dashcam footage and overlay road segmentation masks
4. **Evaluate metrics** — mAP50, mAP50-95, Mask Precision, Mask Recall
5. **Explore multi-class extension** — add lane markings, sidewalk, and non-drivable areas
6. **Benchmark latency** — measure FPS at inference on 1080p input

---

## 10. References

1. Ultralytics YOLO11 — [docs.ultralytics.com](https://docs.ultralytics.com)
2. IDD Dataset — Varma et al., "IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments", WACV 2019
3. PyTorch — [pytorch.org](https://pytorch.org)
4. Git LFS — [git-lfs.github.com](https://git-lfs.github.com)
