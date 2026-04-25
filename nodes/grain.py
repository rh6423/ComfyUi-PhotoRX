"""
GrainRX: Physically-based film grain rendering for ComfyUI.

Uses the pixel-wise Boolean model to synthesize realistic film grain.
Supports both Monte Carlo simulation and fast analytical approximation.
"""

import torch
import numpy as np
from typing import Tuple, Any, Optional

# Import from core module (vendored in this package)
from ..core.profiles import get_profile, list_profiles, PROFILES
from ..core.renderer_fast import render_grayscale_fast, render_color_fast


def _torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to numpy array.

    Handles both single images and batches. Preserves device efficiency by
    only moving to CPU when necessary for numpy conversion.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor in [0, 1] range. Shape: (B, H, W, C) or (H, W, C).

    Returns
    -------
    np.ndarray
        Numpy array with shape (B, H, W, C) - always returns batch dimension.
    """
    # Ensure tensor has batch dimension
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    # Move to CPU and convert to numpy
    np_array = tensor.cpu().numpy()

    return np_array


def _numpy_to_torch(array: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert a numpy array back to PyTorch tensor.

    Parameters
    ----------
    array : np.ndarray
        Input array in [0, 1] range. Shape: (B, H, W, C).
    device : torch.device, optional
        Target device. If None, uses CPU.

    Returns
    -------
    torch.Tensor
        Tensor in [0, 1] float32 range.
    """
    tensor = torch.from_numpy(array.astype(np.float32))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _process_single_image(image_np: np.ndarray, profile_name: str,
                          strength: float, black_white: bool,
                          use_fast: bool, filter_sigma: float,
                          zoom: float, seed: int) -> np.ndarray:
    """
    Process a single image with film grain.

    Parameters
    ----------
    image_np : np.ndarray
        Single image in [0, 1] range, shape (H, W, C).
    profile_name : str
        Film profile key.
    strength : float
        Grain strength multiplier [0.0, 2.0].
    black_white : bool
        If True, convert to grayscale before applying grain.
    use_fast : bool
        Use fast analytical renderer if True, Monte Carlo otherwise.
    filter_sigma : float
        Gaussian filter sigma for grain smoothing.
    zoom : float
        Output zoom factor (>= 1.0).
    seed : int
        Random seed (32-bit).

    Returns
    -------
    np.ndarray
        Processed image in [0, 1] range.
    """
    profile = get_profile(profile_name)

    # Handle alpha channel - extract RGB if present
    if image_np.shape[2] == 4:
        rgb_image = image_np[:, :, :3].copy()
        alpha_channel = image_np[:, :, 3:4].copy()
    else:
        rgb_image = image_np.copy()
        alpha_channel = None

    # Convert to uint8 for rendering (required by core module)
    # Note: This is a lossy conversion - documented limitation
    image_uint8 = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

    # Apply grain based on mode
    if black_white or not profile.color:
        # Grayscale rendering path
        # Convert RGB to luminance using Rec. 709 weights
        gray = np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
        gray_uint8 = np.clip(gray * 255, 0, 255).astype(np.uint8)

        # Scale grain parameters by strength
        mu_r = profile.mu_r * strength
        sigma_r = profile.sigma_r * strength

        if use_fast:
            result = render_grayscale_fast(
                gray_uint8, mu_r, sigma_r,
                filter_sigma=filter_sigma, zoom=zoom, seed=seed
            )
        else:
            from ..core.renderer import render_grayscale
            result = render_grayscale(
                gray_uint8, mu_r, sigma_r,
                filter_sigma=filter_sigma, n_mc=100, zoom=zoom, seed=seed
            )

        # Convert back to float [0, 1] and stack to RGB
        result_float = result.astype(np.float32) / 255.0
        result_rgb = np.stack([result_float] * 3, axis=-1)
    else:
        # Color rendering path
        channel_mu_r = [m * strength for m in profile.channel_mu_r]
        channel_sigma_r = [s * strength for s in profile.channel_sigma_r]

        if use_fast:
            result = render_color_fast(
                image_uint8, channel_mu_r, channel_sigma_r,
                filter_sigma=filter_sigma, zoom=zoom, seed=seed
            )
        else:
            from ..core.renderer import render_color
            result = render_color(
                image_uint8, channel_mu_r, channel_sigma_r,
                filter_sigma=filter_sigma, n_mc=100, zoom=zoom, seed=seed
            )

        result_float = result.astype(np.float32) / 255.0
        result_rgb = result_float

    # Restore alpha channel if present
    if alpha_channel is not None:
        # Resize alpha to match output if zoom != 1
        if zoom != 1.0:
            from PIL import Image as PILImage
            # Squeeze to 2D for PIL (alpha_channel is shape H,W,1)
            alpha_2d = np.clip(alpha_channel[..., 0] * 255, 0, 255).astype(np.uint8)
            alpha_pil = PILImage.fromarray(alpha_2d, mode='L')
            # Resize to actual output size (result_rgb is already zoomed)
            alpha_pil = alpha_pil.resize(
                (result_rgb.shape[1], result_rgb.shape[0]),
                PILImage.BILINEAR,
            )
            alpha_channel = np.array(alpha_pil).astype(np.float32)[..., None] / 255.0

        result_rgb = np.concatenate([result_rgb, alpha_channel], axis=-1)

    return result_rgb


class GrainRX:
    """
    GrainRX - Simple film grain node with preset selection.

    Provides a simple interface for applying film grain using predefined
    film stock profiles. Good for quick results without fine-tuning.

    Inputs:
        image: Input image tensor (B, H, W, C) in [0, 1] range
        profile: Film stock preset (default: portra400)
        strength: Grain intensity multiplier (default: 1.0)
        black_white: Convert to B&W before grain (default: False)
        seed: Random seed for reproducibility (default: 42)

    Outputs:
        image: Image with applied film grain
    """

    FUNCTION = "execute"
    CATEGORY = "PhotoRX/Effects"
    DESCRIPTION = "Apply physically-based film grain to images using preset film stocks. Supports both fast analytical rendering and Monte Carlo simulation."
    SEARCH_ALIASES = ["grain", "film grain", "noise", "film effect", "grainy"]
    OUTPUT_TOOLTIPS = ("Image with film grain applied",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "profile": (sorted(PROFILES.keys()), {"default": "portra400"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "black_white": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF}),
            },
            "optional": {
                "use_fast": ("BOOLEAN", {"default": True}),
                "filter_sigma": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 3.0, "step": 0.1}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Return a hash of all relevant inputs for proper cache invalidation.
        When any input changes, the cache should be invalidated.
        """
        import hashlib
        cache_key = "|".join([
            str(kwargs.get("profile", "portra400")),
            str(kwargs.get("strength", 1.0)),
            str(kwargs.get("black_white", False)),
            str(kwargs.get("seed", 42)),
            str(kwargs.get("use_fast", True)),
            str(kwargs.get("filter_sigma", 0.8)),
            str(kwargs.get("zoom", 1.0)),
        ])
        hash_val = int(hashlib.md5(cache_key.encode()).hexdigest(), 16)
        return hash_val / (16**32)

    def execute(self, image: torch.Tensor, profile: str, strength: float,
                black_white: bool, seed: int,
                use_fast: bool = True, filter_sigma: float = 0.8, zoom: float = 1.0) -> Tuple[torch.Tensor]:
        """
        Apply film grain to input image using simple settings.

        Returns tuple as required by ComfyUI (not dict!).
        """
        # Input validation
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected IMAGE tensor, got {type(image)}")
        if image.dim() not in (3, 4):
            raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")
        if not torch.is_floating_point(image):
            raise TypeError("Image must be floating point tensor")

        # Clamp seed to 32-bit for numpy compatibility
        seed = int(seed) & 0x7FFFFFFF

        # Get original device for output
        device = image.device

        # Convert to numpy (handles batch dimension)
        images_np = _torch_to_numpy(image)

        # Process each image in the batch
        results = []
        for i, img_np in enumerate(images_np):
            # Vary seed per batch item for video/frame sequences
            batch_seed = (seed + i * 0x1337CAFE) & 0xFFFFFFFF
            result = _process_single_image(
                img_np, profile, strength, black_white,
                use_fast, filter_sigma, zoom, batch_seed
            )
            results.append(result)

        # Stack results back into batch tensor
        result_batch = np.stack(results, axis=0)

        # Convert back to tensor on original device
        result_tensor = _numpy_to_torch(result_batch, device)

        return (result_tensor,)


class GrainRX_Advanced:
    """
    GrainRX-Advanced - Full control over all grain parameters.

    Provides complete control over all grain rendering parameters including
    per-channel settings, filter characteristics, and output scaling.

    Inputs:
        image: Input image tensor (B, H, W, C) in [0, 1] range
        profile: Film stock preset (optional, default: portra400)
        strength: Grain intensity multiplier (default: 1.0)
        black_white: Convert to B&W before grain (default: False)
        use_fast: Use fast analytical renderer (default: True)
        filter_sigma: Gaussian filter sigma (default: 0.8)
        zoom: Output zoom factor >= 1.0 (default: 1.0)
        seed: Random seed for reproducibility (default: 42)

    Outputs:
        image: Image with applied film grain
    """

    FUNCTION = "execute"
    CATEGORY = "PhotoRX/Effects"
    DESCRIPTION = "Advanced film grain node with full control over all rendering parameters including filter characteristics and output scaling."
    SEARCH_ALIASES = ["grain advanced", "film grain pro", "grain settings"]
    OUTPUT_TOOLTIPS = ("Image with film grain applied",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "profile": (sorted(PROFILES.keys()), {"default": "portra400"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "black_white": ("BOOLEAN", {"default": False}),
                "use_fast": ("BOOLEAN", {"default": True}),
                "filter_sigma": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 3.0, "step": 0.1}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7FFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Return a hash of all relevant inputs for proper cache invalidation.
        When any input changes, the cache should be invalidated.
        """
        import hashlib
        cache_key = "|".join([
            str(kwargs.get("profile", "portra400")),
            str(kwargs.get("strength", 1.0)),
            str(kwargs.get("black_white", False)),
            str(kwargs.get("seed", 42)),
            str(kwargs.get("use_fast", True)),
            str(kwargs.get("filter_sigma", 0.8)),
            str(kwargs.get("zoom", 1.0)),
        ])
        hash_val = int(hashlib.md5(cache_key.encode()).hexdigest(), 16)
        return hash_val / (16**32)

    def execute(self, image: torch.Tensor, profile: str, strength: float,
                black_white: bool, use_fast: bool, filter_sigma: float,
                zoom: float, seed: int) -> Tuple[torch.Tensor]:
        """
        Apply film grain with advanced parameter control.

        Returns tuple as required by ComfyUI (not dict!).
        """
        # Input validation
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected IMAGE tensor, got {type(image)}")
        if image.dim() not in (3, 4):
            raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")
        if not torch.is_floating_point(image):
            raise TypeError("Image must be floating point tensor")

        # Clamp seed to 32-bit for numpy compatibility
        seed = int(seed) & 0x7FFFFFFF

        # Get original device for output
        device = image.device

        # Convert to numpy (handles batch dimension)
        images_np = _torch_to_numpy(image)

        # Process each image in the batch
        results = []
        for i, img_np in enumerate(images_np):
            # Vary seed per batch item for video/frame sequences
            batch_seed = (seed + i * 0x1337CAFE) & 0xFFFFFFFF
            result = _process_single_image(
                img_np, profile, strength, black_white,
                use_fast, filter_sigma, zoom, batch_seed
            )
            results.append(result)

        # Stack results back into batch tensor
        result_batch = np.stack(results, axis=0)

        # Convert back to tensor on original device
        result_tensor = _numpy_to_torch(result_batch, device)

        return (result_tensor,)


class GrainRX_ProfileInfo:
    """
    GrainRX-ProfileInfo - Display information about a film profile.

    Useful for exploring available presets and their characteristics.

    Inputs:
        profile: Film stock preset to query

    Outputs:
        info: String with profile information
    """

    FUNCTION = "execute"
    CATEGORY = "PhotoRX/Utils"
    DESCRIPTION = "Display detailed information about a selected film profile including grain parameters and description."
    SEARCH_ALIASES = ["profile info", "film info", "preset info"]
    OUTPUT_TOOLTIPS = ("Profile information as formatted string",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "profile": (sorted(PROFILES.keys()), {"default": "portra400"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_NODE = False

    def execute(self, profile: str) -> Tuple[str]:
        """Return formatted profile information as string."""
        try:
            p = get_profile(profile)
            info_lines = [
                f"Profile: {p.name}",
                f"Type: {'Color' if p.color else 'Black & White'}",
                f"mu_r: {p.mu_r:.4f}",
                f"sigma_r: {p.sigma_r:.4f}",
                f"filter_sigma: {p.filter_sigma:.2f}",
            ]
            if p.description:
                info_lines.append(f"Description: {p.description}")
            return ("\n".join(info_lines),)
        except KeyError as e:
            return (f"Error: {e}",)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "GrainRX": GrainRX,
    "GrainRX_Advanced": GrainRX_Advanced,
    "GrainRX_ProfileInfo": GrainRX_ProfileInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrainRX": "GrainRX",
    "GrainRX_Advanced": "GrainRX-Advanced",
    "GrainRX_ProfileInfo": "GrainRX-ProfileInfo",
}