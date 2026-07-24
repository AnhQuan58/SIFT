"""
lib_step1.py
Reference "library" implementation of Step 1 (scale-space extrema detection).

This is the version the project proposal calls "lib": everything is done with
OpenCV, with no custom CUDA/NumPy code. It exists to (a) give the rest of the
pipeline and the web app something to run against from day one, and (b) serve
as the ground-truth Step 1 output that the future cpu / gpu_v1 / gpu_v2 / gpu_v3
backends will be checked against (matching keypoint count/location and pyramid
values, per the correctness plan in the proposal).

Two things happen here, deliberately kept separate:
  1. Gaussian & DoG pyramid construction, done explicitly with cv2.GaussianBlur
     so it can be displayed/inspected (OpenCV's cv2.SIFT does not expose its
     internal pyramid). This is the part later backends will replace with
     custom CUDA kernels.
  2. Candidate keypoint detection, done with cv2.SIFT's built-in detector.
     OpenCV does not expose a separate "find extrema in a pyramid I hand you"
     entry point, so for the lib backend this is the pragmatic stand-in.
     (cpu/gpu backends will instead search the pyramid built in step (1) for
     local extrema themselves, which is the real point of the project.)
"""

from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np

from ..base import Step1Base, Step1Result
from ..registry import register_step1


@register_step1("lib")
class LibStep1(Step1Base):
    """OpenCV-only reference implementation of Step 1."""

    def __init__(self, n_octaves: int = 4, n_scales: int = 5, sigma0: float = 1.6):
        """
        Args:
            n_octaves: number of octaves in the Gaussian/DoG pyramid.
            n_scales: number of blurred images per octave (>=2, so DoG has
                n_scales - 1 images per octave).
            sigma0: base sigma for the first blur of each octave.
        """
        self.n_octaves = n_octaves
        self.n_scales = n_scales
        self.sigma0 = sigma0
        # Used only for candidate keypoint detection (see module docstring).
        self._detector = cv2.SIFT_create()

    def process(self, image_bgr: np.ndarray) -> Step1Result:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        t0 = time.perf_counter()
        gaussian_pyramid, dog_pyramid = self._build_pyramids(gray)
        t1 = time.perf_counter()
        keypoints = self._detector.detect(image_bgr, None)
        t2 = time.perf_counter()

        return Step1Result(
            keypoints=keypoints,
            descriptors=None,  # steps 2-4 (library) fill this in downstream
            gaussian_pyramid=gaussian_pyramid,
            dog_pyramid=dog_pyramid,
            timings={
                "pyramid_ms": (t1 - t0) * 1000.0,
                "detect_ms": (t2 - t1) * 1000.0,
            },
        )

    def _build_pyramids(
        self, gray: np.ndarray
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        """Build the Gaussian pyramid and Difference-of-Gaussians pyramid.

        Mirrors the CPU reference loop from the proposal:
            for octave in range(n_octaves):
                for scale in range(n_scales):
                    pyramid[octave][scale] = gaussian_filter(image, sigma[scale])
                image = downsample(pyramid[octave][-1])
        """
        gaussian_pyramid: List[List[np.ndarray]] = []
        dog_pyramid: List[List[np.ndarray]] = []

        # Standard SIFT inter-scale factor so that after n_scales-1 steps
        # sigma has doubled, matching one octave.
        k = 2.0 ** (1.0 / (self.n_scales - 1))
        image = gray

        for _octave in range(self.n_octaves):
            octave_images: List[np.ndarray] = []
            sigma = self.sigma0
            for _scale in range(self.n_scales):
                blurred = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma)
                octave_images.append(blurred)
                sigma *= k

            gaussian_pyramid.append(octave_images)
            dog_pyramid.append(
                [
                    octave_images[i + 1] - octave_images[i]
                    for i in range(len(octave_images) - 1)
                ]
            )

            # Downsample by 2 for the next octave, guard against images
            # that have shrunk too small to halve further.
            h, w = octave_images[-1].shape[:2]
            if h < 2 or w < 2:
                break
            image = cv2.resize(
                octave_images[-1], (w // 2, h // 2), interpolation=cv2.INTER_NEAREST
            )

        return gaussian_pyramid, dog_pyramid
