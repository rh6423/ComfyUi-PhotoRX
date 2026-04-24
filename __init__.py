"""
PhotoRX - Film Grain Custom Nodes for ComfyUI

A physics-based film grain synthesis plugin using the inhomogeneous Boolean model.
Provides authentic film grain emulation with profiles for popular film stocks.

Installation:
    Place this directory in ComfyUI/custom_nodes/ComfyUi-PhotoRX/

Usage:
    Restart ComfyUI after installation. Nodes will appear under "PhotoRX" category.

Nodes:
    - Film Grain (Basic): Simple interface, just select a film stock
    - Film Grain (Advanced): Full parameter control with renderer selection
    - Profile Info: Helper node to view film profile details

Film Profiles Include:
    B&W: Tri-X 400, HP5 Plus, T-Max series, Delta series, Pan F Plus, Acros
    Color: Portra series, Ektar 100, Superia 400, Gold 200, Ultramax 400

For more information, see README.md or visit the GrainRX documentation.
"""

from .nodes import (
    PhotoRXFilmGrainBasic,
    PhotoRXFilmGrainAdvanced,
    PhotoRXProfileInfo,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

# Required by ComfyUI for node registration
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

# Plugin metadata (optional, for compatibility with some managers)
PLUGIN_NAME = "PhotoRX Film Grain"
PLUGIN_VERSION = "1.0.0"
PLUGIN_DESCRIPTION = "Physics-based film grain synthesis using the Boolean model"


def get_node_class_mappings():
    """Return the node class mappings dictionary."""
    return NODE_CLASS_MAPPINGS


def get_node_display_name_mappings():
    """Return the display name mappings dictionary."""
    return NODE_DISPLAY_NAME_MAPPINGS