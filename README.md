# Project Proposal

## GPU-Accelerated Gaussian Pyramid Construction for SIFT Feature Detection

| Field      | Info                           |
| ---------- | ------------------------------ |
| **Course** | Applied Parallel Programming   |
| **Track**  | Custom Topic — Computer Vision |

---

## 1. Problem Statement

SIFT (Scale-Invariant Feature Transform) is a foundational computer vision algorithm used in panorama stitching, augmented reality, robot localization, and 3D reconstruction. Its core computational step — building a **Gaussian Pyramid** — requires applying Gaussian blur repeatedly across 4 octaves × 5 scales (20+ convolutions per image). On CPU, this serial processing bottleneck makes SIFT unsuitable for real-time or batch applications.

This project implements and optimizes the Gaussian Pyramid construction step using custom CUDA kernels written in Numba. Only the bottleneck step is GPU-accelerated (Partial GPU Principle); keypoint detection and descriptor extraction remain in OpenCV. The project demonstrates three distinct optimization levels: naive global memory → shared memory tiling → fused DoG kernel.

### Dataset

**Primary:** DIV2K — high-resolution image dataset widely used in image processing research, containing 1,000 images at up to 2K resolution (average 2040×1356), split into 800 training and 100 validation images.
Download: https://data.vision.ee.ethz.ch/cvl/DIV2K/

**For quick iteration:** Synthetic images generated via `numpy.random.randn(H, W)` — no download required, deterministic results, controllable size. Both datasets are used in benchmarks.

### Correctness Reference

GPU output is verified against `scipy.ndimage.gaussian_filter` (CPU reference) with MAE < 1e-3 per pixel across all pyramid layers. End-to-end keypoint count is verified against `cv2.SIFT_create()` within ±10% tolerance.

---

## 2. Repository Structure

```
sift-gpu/
├── README.md               ← project title, team, one-line description
├── requirements.txt        ← numpy, numba, cupy-cuda12x, opencv-python, scipy
├── src/
│   ├── cpu_baseline.py     ← must run: python src/cpu_baseline.py
│   ├── gpu_v1_naive.py     ← naive Numba kernel
│   ├── gpu_v2_shared.py    ← shared memory tiling
│   └── gpu_v3_fused.py     ← fused DoG kernel
├── benchmarks/
│   └── benchmark_all.py    ← full speedup table
└── tests/
    └── test_correctness.py ← must pass: pytest tests/
```

---
