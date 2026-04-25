"""
PhotoRX Node Pack - Aggregates all node exports.

This module combines GrainRX (film grain rendering) and SizeRX (dimension calculator)
into a unified node pack for ComfyUI.
"""

from .grain import (
    GrainRX,
    GrainRX_Advanced,
    GrainRX_ProfileInfo,
    NODE_CLASS_MAPPINGS as GRAIN_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as GRAIN_NODE_DISPLAY_NAME_MAPPINGS,
)

from .size import (
    SizeRX,
    NODE_CLASS_MAPPINGS as SIZE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SIZE_NODE_DISPLAY_NAME_MAPPINGS,
)

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **GRAIN_NODE_CLASS_MAPPINGS,
    **SIZE_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **GRAIN_NODE_DISPLAY_NAME_MAPPINGS,
    **SIZE_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = [
    "GrainRX",
    "GrainRX_Advanced",
    "GrainRX_ProfileInfo",
    "SizeRX",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]