"""
PhotoRX Node Pack - Film grain rendering and utilities for ComfyUI.

A unified node pack containing:
- GrainRX: Physically-based film grain renderer using the pixel-wise Boolean model
- SizeRX: Image dimension calculator for aspect ratios and megapixels

Author: rh6423
License: Apache 2.0
"""

from .nodes import (
    GrainRX,
    GrainRX_Advanced,
    GrainRX_ProfileInfo,
    SizeRX,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__version__ = "1.0.0"
__author__ = "rh6423"

# ComfyUI requires these exports for node registration
__all__ = [
    "GrainRX",
    "GrainRX_Advanced",
    "GrainRX_ProfileInfo",
    "SizeRX",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]