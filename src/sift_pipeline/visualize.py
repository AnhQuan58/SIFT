"""
visualize.py
Small, dependency-free drawing helpers shared by the web app and any offline
scripts/notebooks. Kept separate from pipeline.py so visualization never
affects timing measurements of the actual algorithm.
"""

from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np


def draw_keypoints(image_bgr: np.ndarray, keypoints: Sequence[cv2.KeyPoint]) -> np.ndarray:
    """Return a copy of image_bgr with SIFT keypoints (position, scale,
    orientation) drawn as rich circles+direction ticks."""
    return cv2.drawKeypoints(
        image_bgr,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_hud(
    image_bgr: np.ndarray,
    fps: float,
    backend_name: str,
    n_keypoints: int,
    total_ms: float,
) -> np.ndarray:
    """Overlay an FPS / backend / keypoint-count readout in the top-left
    corner. Draws on a copy, does not mutate the input."""
    out = image_bgr.copy()
    lines = [
        f"FPS: {fps:5.1f}",
        f"step1: {backend_name}",
        f"keypoints: {n_keypoints}",
        f"frame: {total_ms:5.1f} ms",
    ]
    x, y = 10, 24
    for line in lines:
        cv2.putText(
            out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 1, cv2.LINE_AA
        )
        y += 22
    return out


def pyramid_montage(pyramid: List[List[np.ndarray]], max_cols: int = 5) -> np.ndarray:
    """Tile a Gaussian or DoG pyramid (list of octaves, each a list of
    images) into a single uint8 grayscale image for quick visual inspection
    (e.g. in a notebook). Not used by the real-time web app (too slow to do
    every frame), but handy for debugging/report figures."""
    if not pyramid:
        return np.zeros((10, 10), dtype=np.uint8)

    def to_u8(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        if hi - lo < 1e-6:
            return np.zeros_like(img, dtype=np.uint8)
        return ((img - lo) / (hi - lo) * 255.0).astype(np.uint8)

    rows = []
    for octave_images in pyramid:
        cols = [to_u8(im) for im in octave_images[:max_cols]]
        h = cols[0].shape[0]
        cols = [cv2.resize(c, (int(c.shape[1] * h / c.shape[0]), h)) for c in cols]
        rows.append(np.hstack(cols))

    max_w = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1], cv2.BORDER_CONSTANT, value=0)
        for r in rows
    ]
    return np.vstack(rows)
