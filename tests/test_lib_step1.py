"""
Basic sanity tests for the "lib" Step 1 backend and the end-to-end pipeline.
Run with:  pytest tests/
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sift_pipeline import SiftPipeline  # noqa: E402
from sift_pipeline.steps.lib_step1 import LibStep1  # noqa: E402


def _synthetic_image(h=256, w=256):
    """Deterministic synthetic BGR image with real structure (concentric
    circles), so SIFT has something to detect -- pure noise or a flat image
    both legitimately yield zero keypoints."""
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    pattern = (np.sin(r / 6.0) * 127 + 128).astype(np.uint8)
    return np.stack([pattern, pattern, pattern], axis=-1)


def test_pyramid_shape():
    step1 = LibStep1(n_octaves=3, n_scales=5)
    image = _synthetic_image()
    result = step1.process(image)

    assert len(result.gaussian_pyramid) == 3
    assert all(len(oct_imgs) == 5 for oct_imgs in result.gaussian_pyramid)
    assert all(len(oct_imgs) == 4 for oct_imgs in result.dog_pyramid)

    # Each octave should be half the spatial size of the previous one.
    h0 = result.gaussian_pyramid[0][0].shape[0]
    h1 = result.gaussian_pyramid[1][0].shape[0]
    assert h1 == h0 // 2


def test_pyramid_blur_increases_with_scale():
    """Higher-sigma images in an octave should be smoother (lower variance
    of the Laplacian is a common smoothness proxy) than the first image."""
    step1 = LibStep1(n_octaves=1, n_scales=5)
    image = _synthetic_image()
    result = step1.process(image)

    def roughness(img):
        return float(np.var(np.gradient(img)))

    first = roughness(result.gaussian_pyramid[0][0])
    last = roughness(result.gaussian_pyramid[0][-1])
    assert last < first


def test_keypoints_found_on_structured_image():
    step1 = LibStep1()
    result = step1.process(_synthetic_image())
    assert len(result.keypoints) > 0


def test_pipeline_produces_descriptors():
    pipeline = SiftPipeline(step1_backend="lib")
    result = pipeline.process_frame(_synthetic_image())

    assert result.step1_backend == "lib"
    if len(result.keypoints) > 0:
        assert result.descriptors is not None
        assert result.descriptors.shape[0] == len(result.keypoints)
        assert result.descriptors.shape[1] == 128  # standard SIFT descriptor length


def test_backend_switch():
    pipeline = SiftPipeline(step1_backend="lib")
    assert "lib" in pipeline.available_backends()
    pipeline.set_step1_backend("lib")  # no-op switch should not raise
    assert pipeline.step1_backend == "lib"
