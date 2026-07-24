"""
base.py
Defines the contract every "Step 1" implementation (lib / cpu / gpu_v1 / gpu_v2 / ...)
must follow, so the rest of the SIFT pipeline (localization, orientation,
descriptor extraction) never needs to know which version produced the keypoints.

Step 1 = Scale-space extrema detection:
    input image  ->  Gaussian Pyramid  ->  DoG Pyramid  ->  candidate keypoints

Everything after Step 1 (keypoint refinement, orientation assignment,
descriptor extraction) is intentionally NOT part of this interface -- the
pipeline always completes those stages with a library implementation
(OpenCV's cv2.SIFT), regardless of which Step 1 backend is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Step1Result:
    """Everything Step 1 hands off to the rest of the pipeline / UI."""

    # Candidate keypoints found by this Step 1 backend. Required.
    keypoints: List[cv2.KeyPoint]

    # Optional: descriptors, ONLY if this backend also computed them itself
    # (the reference "lib" backend does not; cv2.SIFT.compute() fills them
    # in later, in the pipeline). Leave as None to let the pipeline complete
    # steps 2-4 with the library.
    descriptors: Optional[np.ndarray] = None

    # Gaussian pyramid, one list of grayscale float32 images per octave.
    # gaussian_pyramid[octave][scale] -> np.ndarray (H, W), float32
    gaussian_pyramid: List[List[np.ndarray]] = field(default_factory=list)

    # Difference-of-Gaussians pyramid, same nested-list shape as above
    # (one fewer scale per octave than the Gaussian pyramid).
    dog_pyramid: List[List[np.ndarray]] = field(default_factory=list)

    # Free-form timing breakdown in milliseconds, e.g. {"pyramid_ms": 4.2,
    # "detect_ms": 1.1}. Used by the web app / benchmarks to report per-stage
    # cost without caring which backend produced it.
    timings: dict = field(default_factory=dict)


class Step1Base(ABC):
    """Abstract interface every Step 1 backend must implement.

    Subclasses register themselves with @register_step1("name") in
    registry.py so the pipeline / web app can switch backends by name at
    runtime, e.g. STEP1_REGISTRY["lib"], STEP1_REGISTRY["gpu_v1"], ...
    """

    #: Short, unique identifier used for registry lookups and the UI dropdown.
    name: str = "base"

    @abstractmethod
    def process(self, image_bgr: np.ndarray) -> Step1Result:
        """Run scale-space extrema detection on a single BGR uint8 frame.

        Args:
            image_bgr: HxWx3 uint8 image straight from the camera / file.

        Returns:
            A Step1Result with at least `keypoints` populated.
        """
        raise NotImplementedError
