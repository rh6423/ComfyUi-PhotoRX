"""
GrainRX Engine - PyTorch-compatible wrapper for ComfyUI integration.

This module provides a clean interface to the GrainRX rendering engine,
handling conversion between PyTorch tensors (ComfyUI format) and NumPy arrays.
"""

import numpy as np
import torch
from PIL import Image

# Import GrainRX core modules
from core.profiles import get_profile, PROFILES
from core.renderer_fast import render_grayscale_fast, render_color_fast
from core.renderer import render_grayscale, render_color, warmup_jit


class GrainEngine:
    """
    Film grain rendering engine for ComfyUI.
    
    Supports both fast analytical renderer and Monte Carlo Boolean model.
    """
    
    # Cache for JIT warmup
    _mc_warmed_up = False
    
    @classmethod
    def get_all_profiles(cls):
        """Return list of all available film profile keys."""
        return sorted(PROFILES.keys())
    
    @classmethod
    def get_profile_info(cls, key):
        """Get detailed information about a film profile."""
        profile = get_profile(key)
        return {
            'key': key,
            'name': profile.name,
            'color': profile.color,
            'mu_r': profile.mu_r,
            'sigma_r': profile.sigma_r,
            'filter_sigma': profile.filter_sigma,
            'channel_mu_r': profile.channel_mu_r,
            'channel_sigma_r': profile.channel_sigma_r,
            'description': profile.description,
        }
    
    @classmethod
    def warmup_mc(cls):
        """Warm up Monte Carlo JIT compilation."""
        if not cls._mc_warmed_up:
            warmup_jit()
            cls._mc_warmed_up = True
    
    @staticmethod
    def _torch_to_numpy(tensor):
        """
        Convert ComfyUI image tensor to NumPy array.
        
        Input: (batch, height, width, channels) float32 [0, 1]
        Output: (height, width, channels) or (height, width) uint8 [0, 255]
        """
        # Take first image from batch
        img = tensor[0].cpu().numpy()
        # Convert from [0, 1] float to [0, 255] uint8
        img = (img * 255).astype(np.uint8)
        return img
    
    @staticmethod
    def _numpy_to_torch(array):
        """
        Convert NumPy array to ComfyUI image tensor.
        
        Input: (height, width, channels) or (height, width) uint8 [0, 255]
        Output: (1, height, width, channels) float32 [0, 1]
        """
        # Ensure 3D array
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        # Convert from [0, 255] uint8 to [0, 1] float32
        img = array.astype(np.float32) / 255.0
        # Add batch dimension
        img = img[np.newaxis, ...]
        # Convert to PyTorch tensor
        return torch.from_numpy(img)
    
    @staticmethod
    def _to_grayscale(rgb_array):
        """Convert RGB array to grayscale using standard luminance weights."""
        return np.dot(rgb_array[..., :3], [0.2126, 0.7152, 0.0722]).astype(np.uint8)
    
    @classmethod
    def apply_grain_fast(cls, image_tensor, profile_key, bw=False, 
                         zoom=1.0, seed=42):
        """
        Apply film grain using the fast analytical renderer.
        
        Args:
            image_tensor: ComfyUI image tensor (batch, H, W, C) float32 [0, 1]
            profile_key: Film profile key (e.g., 'tri-x', 'portra400')
            bw: If True, convert to grayscale before applying grain
            zoom: Output resolution multiplier
            seed: Random seed for reproducibility
            
        Returns:
            ComfyUI image tensor with grain applied
        """
        profile = get_profile(profile_key)
        
        # Convert to NumPy
        img_np = cls._torch_to_numpy(image_tensor)
        
        if bw:
            # Convert to grayscale
            gray = cls._to_grayscale(img_np)
            result = render_grayscale_fast(gray, profile.mu_r, profile.sigma_r,
                                          zoom=zoom, seed=seed)
            # Convert back to RGB for ComfyUI
            result = np.stack([result] * 3, axis=-1)
        else:
            if profile.color:
                result = render_color_fast(img_np, profile.channel_mu_r, 
                                          profile.channel_sigma_r,
                                          zoom=zoom, seed=seed)
            else:
                # B&W profile on color image - apply to luminance
                gray = cls._to_grayscale(img_np)
                result_gray = render_grayscale_fast(gray, profile.mu_r, 
                                                   profile.sigma_r,
                                                   zoom=zoom, seed=seed)
                result = np.stack([result_gray] * 3, axis=-1)
        
        return cls._numpy_to_torch(result)
    
    @classmethod
    def apply_grain_mc(cls, image_tensor, profile_key, bw=False,
                       n_mc=100, zoom=1.0, seed=42):
        """
        Apply film grain using the Monte Carlo Boolean model renderer.
        
        Args:
            image_tensor: ComfyUI image tensor (batch, H, W, C) float32 [0, 1]
            profile_key: Film profile key (e.g., 'tri-x', 'portra400')
            bw: If True, convert to grayscale before applying grain
            n_mc: Number of Monte Carlo samples per pixel
            zoom: Output resolution multiplier
            seed: Random seed for reproducibility
            
        Returns:
            ComfyUI image tensor with grain applied
        """
        # Warm up JIT if needed
        cls.warmup_mc()
        
        profile = get_profile(profile_key)
        
        # Convert to NumPy
        img_np = cls._torch_to_numpy(image_tensor)
        
        if bw:
            # Convert to grayscale
            gray = cls._to_grayscale(img_np)
            result = render_grayscale(gray, profile.mu_r, profile.sigma_r,
                                     n_mc=n_mc, zoom=zoom, seed=seed)
            # Convert back to RGB for ComfyUI
            result = np.stack([result] * 3, axis=-1)
        else:
            if profile.color:
                result = render_color(img_np, profile.channel_mu_r,
                                     profile.channel_sigma_r,
                                     n_mc=n_mc, zoom=zoom, seed=seed)
            else:
                # B&W profile on color image - apply to luminance
                gray = cls._to_grayscale(img_np)
                result_gray = render_grayscale(gray, profile.mu_r, 
                                              profile.sigma_r,
                                              n_mc=n_mc, zoom=zoom, seed=seed)
                result = np.stack([result_gray] * 3, axis=-1)
        
        return cls._numpy_to_torch(result)
    
    @classmethod
    def apply_grain_advanced(cls, image_tensor, profile_key, renderer='fast',
                             mu_r=None, sigma_r=None, filter_sigma=0.8,
                             n_mc=100, zoom=1.0, seed=42, bw=False,
                             visibility_modulation=False):
        """
        Apply film grain with advanced parameter control.
        
        Args:
            image_tensor: ComfyUI image tensor (batch, H, W, C) float32 [0, 1]
            profile_key: Film profile key (used as base for parameters)
            renderer: 'fast' or 'mc' (Monte Carlo)
            mu_r: Override mean grain radius (uses profile default if None)
            sigma_r: Override grain radius std dev (uses profile default if None)
            filter_sigma: Gaussian filter sigma in output pixels
            n_mc: Monte Carlo samples (only used with renderer='mc')
            zoom: Output resolution multiplier
            seed: Random seed for reproducibility
            bw: If True, convert to grayscale before applying grain
            visibility_modulation: Apply perceptual visibility modulation
            
        Returns:
            ComfyUI image tensor with grain applied
        """
        profile = get_profile(profile_key)
        
        # Use profile defaults or overrides
        effective_mu_r = mu_r if mu_r is not None else profile.mu_r
        effective_sigma_r = sigma_r if sigma_r is not None else profile.sigma_r
        
        # Convert to NumPy
        img_np = cls._torch_to_numpy(image_tensor)
        
        if renderer == 'mc':
            cls.warmup_mc()
            
            if bw:
                gray = cls._to_grayscale(img_np)
                result = render_grayscale(gray, effective_mu_r, effective_sigma_r,
                                         filter_sigma=filter_sigma, n_mc=n_mc,
                                         zoom=zoom, seed=seed)
                result = np.stack([result] * 3, axis=-1)
            else:
                # For color with custom params, use per-channel averages
                ch_mu_r = [effective_mu_r] * 3
                ch_sigma_r = [effective_sigma_r] * 3
                result = render_color(img_np, ch_mu_r, ch_sigma_r,
                                     filter_sigma=filter_sigma, n_mc=n_mc,
                                     zoom=zoom, seed=seed)
        else:
            # Fast renderer
            if bw:
                gray = cls._to_grayscale(img_np)
                result = render_grayscale_fast(gray, effective_mu_r, 
                                              effective_sigma_r,
                                              filter_sigma=filter_sigma,
                                              zoom=zoom, seed=seed)
                result = np.stack([result] * 3, axis=-1)
            else:
                ch_mu_r = [effective_mu_r] * 3
                ch_sigma_r = [effective_sigma_r] * 3
                result = render_color_fast(img_np, ch_mu_r, ch_sigma_r,
                                          filter_sigma=filter_sigma,
                                          zoom=zoom, seed=seed)
        
        return cls._numpy_to_torch(result)


# Convenience functions for direct use
def get_profiles_list():
    """Get sorted list of all film profile keys."""
    return GrainEngine.get_all_profiles()

def apply_grain(image_tensor, profile_key='tri-x', bw=False):
    """
    Quick grain application using fast renderer.
    
    Args:
        image_tensor: ComfyUI image tensor
        profile_key: Film profile key
        bw: Convert to grayscale before applying grain
        
    Returns:
        Image tensor with grain applied
    """
    return GrainEngine.apply_grain_fast(image_tensor, profile_key, bw=bw)