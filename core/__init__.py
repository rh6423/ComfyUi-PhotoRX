"""
GrainRX: Physics-based film grain rendering using the Boolean model.

Based on:
  Newson, Faraj, Delon, Galerne. "Realistic Film Grain Rendering."
  Image Processing On Line, 7 (2017), pp. 165-183.

Unlike overlay-based approaches, this synthesizes the image FROM the grain:
the output pixel values emerge from the statistical density of simulated
silver halide crystals, filtered through a Gaussian kernel modeling the
optical enlargement process and human vision.
"""
from .renderer import render_grayscale, render_color, warmup_jit
from .renderer_fast import render_grayscale_fast, render_color_fast
from .profiles import FilmProfile, get_profile, list_profiles, PROFILES
from .postprocess import (
    apply_characteristic_curve,
    apply_visibility_modulation,
    apply_chromatic_aberration,
)

__version__ = "0.2.1"

__all__ = [
    # Monte Carlo renderer (ground-truth quality)
    "render_grayscale",
    "render_color",
    "warmup_jit",
    # Fast analytical renderer (~100-500x faster)
    "render_grayscale_fast",
    "render_color_fast",
    # Profiles and post-processing
    "FilmProfile",
    "get_profile",
    "list_profiles",
    "PROFILES",
    "apply_characteristic_curve",
    "apply_visibility_modulation",
    "apply_chromatic_aberration",
]