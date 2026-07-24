"""
registry.py
A tiny name -> class registry for Step 1 backends.

Add a new backend later (e.g. CPU baseline, GPU V1 naive, GPU V2 tiled) by
creating a new file in sift_pipeline/steps/, subclassing Step1Base, and
decorating it with @register_step1("cpu"). Nothing else in the codebase
(pipeline.py, the web app, the dropdown) needs to change.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Step1Base

STEP1_REGISTRY: Dict[str, Type[Step1Base]] = {}


def register_step1(name: str):
    """Class decorator: register a Step1Base subclass under `name`."""

    def _decorator(cls: Type[Step1Base]):
        if name in STEP1_REGISTRY:
            raise ValueError(f"Step1 backend '{name}' is already registered")
        cls.name = name
        STEP1_REGISTRY[name] = cls
        return cls

    return _decorator


def available_backends() -> list[str]:
    """Names of all currently-registered Step 1 backends, in insertion order."""
    return list(STEP1_REGISTRY.keys())


def create_step1(name: str) -> Step1Base:
    """Instantiate a registered backend by name."""
    if name not in STEP1_REGISTRY:
        raise KeyError(
            f"Unknown Step1 backend '{name}'. Available: {available_backends()}"
        )
    return STEP1_REGISTRY[name]()
