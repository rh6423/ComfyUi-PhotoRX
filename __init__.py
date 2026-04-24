"""
PhotoRX - Film Grain Rendering for ComfyUI

A physically-based film grain renderer using the pixel-wise Boolean model.
Supports both Monte Carlo simulation and fast analytical approximation.

Author: rh6423
License: MIT
"""

from .nodes import (
    PhotoRXFilmGrainBasic,
    PhotoRXFilmGrainAdvanced,
    PhotoRXProfileInfo,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__version__ = "1.0.0"
__author__ = "rh6423"

# ComfyUI requires these exports for node registration
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]