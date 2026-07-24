"""
pipeline.py
Wires the full 4-stage SIFT pipeline together:

    Step 1 (swappable: lib / cpu / gpu_v1 / gpu_v2 / ...)
        scale-space extrema detection -> candidate keypoints
    Steps 2-4 (always library / OpenCV)
        keypoint localization + orientation assignment + descriptor extraction

Whatever Step 1 backend is active, this class is the single place the rest
of the app (web demo, benchmarks, tests) talks to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from . import steps  # noqa: F401  (import triggers backend self-registration)
from .base import Step1Result
from .registry import available_backends, create_step1


@dataclass
class ProcessResult:
    """Full pipeline output for a single frame."""

    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    total_ms: float
    step1_timings: dict = field(default_factory=dict)
    step1_backend: str = ""
    gaussian_pyramid: list = field(default_factory=list)
    dog_pyramid: list = field(default_factory=list)


class SiftPipeline:
    """End-to-end SIFT pipeline with a swappable Step 1 backend."""

    def __init__(self, step1_backend: str = "lib"):
        # cv2.SIFT instance used to complete steps 2-4 (library, always).
        self._sift = cv2.SIFT_create()
        self.step1 = None
        self.step1_backend = None
        self.set_step1_backend(step1_backend)

    @staticmethod
    def available_backends() -> List[str]:
        return available_backends()

    def set_step1_backend(self, name: str) -> None:
        """Swap the Step 1 implementation at runtime (thread-safety is the
        caller's responsibility -- see webapp/app.py for the lock used
        around this call)."""
        self.step1 = create_step1(name)
        self.step1_backend = name

    def process_frame(self, image_bgr: np.ndarray) -> ProcessResult:
        t0 = time.perf_counter()

        result: Step1Result = self.step1.process(image_bgr)

        # Steps 2-4: always completed by the library. If the backend already
        # produced descriptors itself (a future backend might), reuse them;
        # otherwise ask OpenCV to localize/orient/describe the given
        # candidate keypoints.
        if result.descriptors is not None:
            keypoints, descriptors = result.keypoints, result.descriptors
        else:
            keypoints, descriptors = self._sift.compute(image_bgr, result.keypoints)

        total_ms = (time.perf_counter() - t0) * 1000.0

        return ProcessResult(
            keypoints=keypoints if keypoints is not None else [],
            descriptors=descriptors,
            total_ms=total_ms,
            step1_timings=result.timings,
            step1_backend=self.step1_backend,
            gaussian_pyramid=result.gaussian_pyramid,
            dog_pyramid=result.dog_pyramid,
        )
