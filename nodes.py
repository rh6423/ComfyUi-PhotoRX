"""
PhotoRX Custom Nodes for ComfyUI

Film grain synthesis nodes based on the physics-based Boolean model.
Provides both a simple interface and advanced parameter control.
"""

import torch
from .grain_engine import GrainEngine


# ============================================================================
# Basic Film Grain Node
# ============================================================================

class PhotoRXFilmGrainBasic:
    """
    Simple film grain node with minimal parameters.
    
    Just select a film stock and apply authentic grain to your image.
    Uses the fast analytical renderer for quick results.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        profiles = GrainEngine.get_all_profiles()
        
        return {
            "required": {
                "image": ("IMAGE",),
                "film_profile": (profiles, {"default": "tri-x"}),
                "black_white": ("BOOLEAN", {
                    "default": False,
                    "display_name": "Black & White"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "zoom": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.1,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "PhotoRX/Grain"
    DISPLAY_NAME = "Film Grain (Basic)"
    
    def execute(self, image, film_profile, black_white, seed, zoom=1.0):
        result = GrainEngine.apply_grain_fast(
            image_tensor=image,
            profile_key=film_profile,
            bw=black_white,
            zoom=zoom,
            seed=seed
        )
        return {"image": result}


# ============================================================================
# Advanced Film Grain Node
# ============================================================================

class PhotoRXFilmGrainAdvanced:
    """
    Advanced film grain node with full parameter control.
    
    Choose between fast analytical renderer or Monte Carlo Boolean model,
    and fine-tune all grain parameters for precise control.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        profiles = GrainEngine.get_all_profiles()
        
        return {
            "required": {
                "image": ("IMAGE",),
                "film_profile": (profiles, {"default": "tri-x"}),
                "renderer": (["fast", "monte_carlo"], {
                    "default": "fast",
                    "display_name": "Renderer"
                }),
                "black_white": ("BOOLEAN", {
                    "default": False,
                    "display_name": "Black & White"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                # Grain geometry parameters
                "mu_r": ("FLOAT", {
                    "default": 0.07,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.001,
                    "display": "number"
                }),
                "sigma_r": ("FLOAT", {
                    "default": 0.025,
                    "min": 0.0,
                    "max": 0.3,
                    "step": 0.001,
                    "display": "number"
                }),
                "filter_sigma": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "number"
                }),
                # Monte Carlo parameters
                "mc_samples": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
                # Output settings
                "zoom": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.1,
                    "display": "number"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "PhotoRX/Grain"
    DISPLAY_NAME = "Film Grain (Advanced)"
    
    def execute(self, image, film_profile, renderer, black_white, seed,
                mu_r=0.07, sigma_r=0.025, filter_sigma=0.8,
                mc_samples=100, zoom=1.0):
        
        # Map renderer name to internal format
        render_mode = 'mc' if renderer == 'monte_carlo' else 'fast'
        
        result = GrainEngine.apply_grain_advanced(
            image_tensor=image,
            profile_key=film_profile,
            renderer=render_mode,
            mu_r=mu_r,
            sigma_r=sigma_r,
            filter_sigma=filter_sigma,
            n_mc=mc_samples,
            zoom=zoom,
            seed=seed,
            bw=black_white
        )
        
        return {"image": result}


# ============================================================================
# Film Profile Info Node (Helper)
# ============================================================================

class PhotoRXProfileInfo:
    """
    Helper node that displays film profile information.
    
    Useful for exploring available profiles and their parameters.
    Outputs profile data as strings for display in ComfyUI.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        profiles = GrainEngine.get_all_profiles()
        
        return {
            "required": {
                "film_profile": (profiles, {"default": "tri-x"}),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("name", "description", "mu_r", "sigma_r")
    FUNCTION = "execute"
    CATEGORY = "PhotoRX/Utils"
    DISPLAY_NAME = "Profile Info"
    
    def execute(self, film_profile):
        info = GrainEngine.get_profile_info(film_profile)
        return (
            info['name'],
            info['description'],
            info['mu_r'],
            info['sigma_r']
        )


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "PhotoRXFilmGrainBasic": PhotoRXFilmGrainBasic,
    "PhotoRXFilmGrainAdvanced": PhotoRXFilmGrainAdvanced,
    "PhotoRXProfileInfo": PhotoRXProfileInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoRXFilmGrainBasic": "Film Grain (Basic)",
    "PhotoRXFilmGrainAdvanced": "Film Grain (Advanced)",
    "PhotoRXProfileInfo": "Profile Info",
}


# ============================================================================
# Web UI Extensions (Optional - for custom widgets)
# ============================================================================

WEB_DIRECTORY = "./web"

__all__ = [
    "PhotoRXFilmGrainBasic",
    "PhotoRXFilmGrainAdvanced", 
    "PhotoRXProfileInfo",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]