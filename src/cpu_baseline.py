"""
CPU Baseline — Gaussian Pyramid construction (single-threaded)
================================================================
Reference implementation used to:
  1. Establish the CPU timing baseline (see proposal Section 2).
  2. Provide the "ground truth" pyramid that every GPU version is
     checked against (MAE < 1e-3 per pixel).

Run directly:  python src/cpu_baseline.py
"""

import time
import os
import glob
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# ----------------------------------------------------------------------
# Pyramid configuration (matches proposal: 4 octaves x 5 scales)
# ----------------------------------------------------------------------
N_OCTAVES = 4
N_SCALES = 5
SIGMA_BASE = 1.6
K = 2 ** (1.0 / (N_SCALES - 3))  # standard SIFT scale multiplier


def sigmas_for_octave():
    """Per-scale sigma values within a single octave (SIFT convention)."""
    return [SIGMA_BASE * (K ** s) for s in range(N_SCALES)]


def build_gaussian_pyramid_cpu(image: np.ndarray):
    """
    Build a 4-octave x 5-scale Gaussian pyramid.

    Parameters
    ----------
    image : np.ndarray, float32, shape (H, W)

    Returns
    -------
    list[list[np.ndarray]]  pyramid[octave][scale]
    """
    assert image.dtype == np.float32, "expected float32 input"
    sigmas = sigmas_for_octave()

    pyramid = []
    current = image
    for octave in range(N_OCTAVES):
        octave_images = []
        for sigma in sigmas:
            blurred = gaussian_filter(current, sigma=sigma, mode="reflect")
            octave_images.append(blurred.astype(np.float32))
        pyramid.append(octave_images)
        # downsample using the 3rd scale image of this octave (SIFT convention)
        current = octave_images[N_SCALES - 3][::2, ::2]
    return pyramid


def build_dog_pyramid_cpu(gauss_pyramid):
    """Difference-of-Gaussians pyramid, derived from the Gaussian pyramid."""
    dog_pyramid = []
    for octave_images in gauss_pyramid:
        dog_images = [
            octave_images[i + 1] - octave_images[i]
            for i in range(len(octave_images) - 1)
        ]
        dog_pyramid.append(dog_images)
    return dog_pyramid


def _time_run(image, n_runs=3):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        pyr = build_gaussian_pyramid_cpu(image)
        times.append(time.perf_counter() - t0)
    return pyr, times


if __name__ == "__main__":
    div2k_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "DIV2K_train_HR"))
    image_paths = glob.glob(os.path.join(div2k_dir, "*.png"))
    
    if not image_paths:
        print(f"No images found in {div2k_dir}. Falling back to random noise.")
        H, W = 1356, 2040
        rng = np.random.default_rng(0)
        image = rng.standard_normal((H, W), dtype=np.float32)
        
        pyramid, times = _time_run(image, n_runs=3)
        dog = build_dog_pyramid_cpu(pyramid)
        avg_ms = 1000 * sum(times) / len(times)
        print(f"Image size: {W} x {H} (float32)")
        print(f"Pyramid: {N_OCTAVES} octaves x {N_SCALES} scales")
        print(f"Runs: {[f'{t*1000:.1f} ms' for t in times]}")
        print(f"Average CPU time: {avg_ms:.1f} ms/image")
        print(f"DoG layers per octave: {len(dog[0])}")
    else:
        # Sort to ensure consistent order
        image_paths = sorted(image_paths)
        selected_paths = image_paths[:10]
        
        print(f"Found {len(image_paths)} images, benchmarking on the first {len(selected_paths)} images...")
        
        all_times = []
        for i, img_path in enumerate(selected_paths, 1):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load {img_path}")
                continue
            image = img.astype(np.float32)
            H, W = image.shape

            pyramid, times = _time_run(image, n_runs=3)
            all_times.extend(times)
            dog = build_dog_pyramid_cpu(pyramid)
            
            avg_ms = 1000 * sum(times) / len(times)
            print(f"[{i}/{len(selected_paths)}] {os.path.basename(img_path)} ({W}x{H}) - Avg CPU time: {avg_ms:.1f} ms")

        if all_times:
            total_avg_ms = 1000 * sum(all_times) / len(all_times)
            print("-" * 40)
            print(f"Pyramid: {N_OCTAVES} octaves x {N_SCALES} scales")
            print(f"DoG layers per octave: {len(dog[0])}")
            print(f"Overall Average CPU time: {total_avg_ms:.1f} ms/image")