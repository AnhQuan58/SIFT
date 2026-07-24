# SIFT-GPU — pluggable Step 1 + live webcam demo

Implements SIFT as four stages, where **Step 1 (scale-space extrema
detection: Gaussian Pyramid → DoG Pyramid → candidate keypoints)** is a
swappable backend, and **Steps 2–4 (keypoint localization, orientation
assignment, descriptor extraction)** always run through OpenCV.

Only the **`lib`** backend is implemented right now, as requested. The
architecture is built so `cpu`, `gpu_v1`, `gpu_v2`, `gpu_v3`, ... can be
dropped in later without touching the pipeline, the tests, or the web app.

```
sift_project/
├── requirements.txt
├── src/sift_pipeline/
│   ├── base.py          # Step1Base interface + Step1Result container
│   ├── registry.py       # name -> backend class registry
│   ├── pipeline.py       # SiftPipeline: runs Step1, then library steps 2-4
│   ├── visualize.py      # keypoint overlay, FPS/HUD text, pyramid montage
│   └── steps/
│       ├── __init__.py    # imports (= registers) every backend module
│       └── lib_step1.py   # the "lib" backend (implemented)
├── webapp/
│   ├── app.py            # Flask app: camera thread + MJPEG streams + API
│   ├── templates/index.html
│   └── static/{style.css, script.js}
└── tests/test_lib_step1.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the tests

```bash
pytest tests/
```

## Run the live webcam demo

```bash
python webapp/app.py
```

Then open **http://127.0.0.1:5000**. The page shows the raw webcam feed on
the left and the SIFT keypoint overlay on the right, with FPS, keypoint
count, and the active Step 1 backend readout underneath. The dropdown in the
top bar switches the Step 1 backend live (only `lib` is available until
more backends are added).

If the camera fails to open, check `CAMERA_INDEX` at the top of
`webapp/app.py` (it defaults to `0`) and make sure no other application is
using the webcam.

## How the "lib" backend works

`LibStep1` (`src/sift_pipeline/steps/lib_step1.py`) does two things:

1. Builds an explicit Gaussian pyramid (`cv2.GaussianBlur` per octave/scale)
   and Difference-of-Gaussians pyramid, following the same
   octaves × scales loop structure as the proposal's CPU baseline. This is
   the part future `cpu`/`gpu_v1`/`gpu_v2`/`gpu_v3` backends will replace
   with custom NumPy/CUDA code, and it's exposed on `Step1Result` for
   inspection/debugging (see `visualize.pyramid_montage`).
2. Detects candidate keypoints with `cv2.SIFT`'s built-in detector, since
   OpenCV doesn't expose a "search extrema in a pyramid I hand you" entry
   point. Future backends will instead search their own pyramid from step
   (1) for local extrema directly — that's the actual point of the project
   and what will be benchmarked/compared against this `lib` reference.

`SiftPipeline.process_frame()` then always completes steps 2–4 by calling
`cv2.SIFT().compute()` on whatever keypoints Step 1 produced, so descriptor
quality/format is identical no matter which Step 1 backend is active.

## Adding a new Step 1 backend later

1. Create `src/sift_pipeline/steps/cpu_step1.py` (or `gpu_v1_step1.py`, ...).
2. Subclass `Step1Base`, implement `process(self, image_bgr) -> Step1Result`,
   and decorate the class with `@register_step1("cpu")`.
3. Uncomment the matching import in `src/sift_pipeline/steps/__init__.py`.

Nothing else changes: the pipeline, the tests, and the web app's dropdown
all pick up the new backend automatically via the registry.
