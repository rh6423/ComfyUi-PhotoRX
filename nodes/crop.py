"""
CropRX: Visual cropping node for ComfyUI.

Provides visual crop bars with locked/free aspect ratio options,
pixel input fields, and draggable crop area - similar to Photoshop's
crop tool but using native ComfyUI components.
"""

import torch
import numpy as np
from typing import Tuple, Any, Optional


class CropRX:
    """
    CropRX - Visual cropping node with aspect ratio controls.

    Features:
    - Free or locked aspect ratio mode
    - Preset aspect ratios (same as SizeRX): 1:1, 4:3, 3:2, 16:9, 2:1, 21:9
    - Visual crop bars on all four sides
    - Pixel width and height input fields that sync with visual crop
    - Draggable crop area in the visual editor
    - Outputs cropped image

    Inputs:
        image: Input image tensor (B, H, W, C) in [0, 1] range
        mode: 'free' or 'locked' aspect ratio mode
        aspect_ratio: Preset aspect ratio (when locked)
        width: Target crop width in pixels
        height: Target crop height in pixels
        x_offset: Horizontal offset from left edge
        y_offset: Vertical offset from top edge

    Outputs:
        image: Cropped image tensor
    """

    # Aspect ratios matching SizeRX
    ASPECT_KEYS = [
        "1:1",      # Square
        "4:3",      # Classic TV, 35mm film frame
        "3:2",      # 35mm photography, DSLR default
        "16:9",     # HD video, modern TV/monitor
        "2:1",      # Ultrawide photography, Instagram
        "21:9",     # CinemaScope, ultrawide monitors
    ]

    FUNCTION = "execute"
    CATEGORY = "PhotoRX/Transform"
    DESCRIPTION = "Visual cropping node with aspect ratio controls. Features crop bars, pixel inputs, and draggable crop area - similar to Photoshop's crop tool."
    SEARCH_ALIASES = ["crop", "cropper", "trim", "slice"]
    OUTPUT_TOOLTIPS = ("Cropped image",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                # Crop mode selector
                "mode": (["free", "locked"], {"default": "locked"}),
                # Aspect ratio preset (used when mode is 'locked')
                "aspect_ratio": (cls.ASPECT_KEYS, {"default": "16:9"}),
                # Target dimensions in pixels
                "width": ("INT", {
                    "default": 512,
                    "min": 8,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 288,
                    "min": 8,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                # Position offsets (for dragging/repositioning)
                "x_offset": ("INT", {
                    "default": 0,
                    "min": -8192,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                "y_offset": ("INT", {
                    "default": 0,
                    "min": -8192,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """Return a hash of all relevant inputs for cache invalidation."""
        import hashlib
        cache_key = "|".join([
            str(kwargs.get("mode", "locked")),
            str(kwargs.get("aspect_ratio", "16:9")),
            str(kwargs.get("width", 512)),
            str(kwargs.get("height", 288)),
            str(kwargs.get("x_offset", 0)),
            str(kwargs.get("y_offset", 0)),
        ])
        hash_val = int(hashlib.md5(cache_key.encode()).hexdigest(), 16)
        return hash_val / (16**32)

    def _parse_aspect_ratio(self, preset: str) -> Tuple[float, float]:
        """Parse aspect ratio string to width and height ratios."""
        if ":" not in preset:
            raise ValueError(f"Invalid preset format '{preset}': expected 'W:H' format")
        parts = preset.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid preset format '{preset}': expected exactly one colon")
        w_ratio, h_ratio = map(int, parts)
        if w_ratio <= 0 or h_ratio <= 0:
            raise ValueError(f"Invalid preset '{preset}': ratios must be positive integers")
        return (float(w_ratio), float(h_ratio))

    def _clamp_crop_bounds(
        self,
        x: int, y: int, width: int, height: int,
        img_width: int, img_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Clamp crop rectangle to valid image bounds.

        Returns clamped (x, y, width, height).
        """
        # Ensure positive dimensions
        width = max(width, 1)
        height = max(height, 1)

        # Clamp x position
        x = max(0, min(x, img_width - 1))
        # Clamp y position
        y = max(0, min(y, img_height - 1))

        # Adjust width/height if crop extends beyond image
        if x + width > img_width:
            width = img_width - x
        if y + height > img_height:
            height = img_height - y

        # Ensure we still have valid dimensions
        width = max(width, 1)
        height = max(height, 1)

        return (x, y, width, height)

    def execute(
        self,
        image: torch.Tensor,
        mode: str,
        aspect_ratio: str,
        width: int,
        height: int,
        x_offset: int,
        y_offset: int
    ) -> Tuple[torch.Tensor]:
        """
        Crop the input image based on specified parameters.

        Args:
            image: Input image tensor (B, H, W, C) in [0, 1] range
            mode: 'free' or 'locked' aspect ratio mode
            aspect_ratio: Preset aspect ratio (when locked)
            width: Target crop width in pixels
            height: Target crop height in pixels
            x_offset: Horizontal offset from left edge
            y_offset: Vertical offset from top edge

        Returns:
            Tuple containing cropped image tensor
        """
        # Input validation
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected IMAGE tensor, got {type(image)}")
        if image.dim() not in (3, 4):
            raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")
        if not torch.is_floating_point(image):
            raise TypeError("Image must be floating point tensor")

        # Get original device for output
        device = image.device

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        batch_size, img_height, img_width, channels = image.shape

        # Parse aspect ratio for locked mode calculations
        w_ratio, h_ratio = self._parse_aspect_ratio(aspect_ratio)

        # Adjust dimensions based on mode
        if mode == "locked":
            # In locked mode, width determines height via aspect ratio
            # Calculate the corresponding height from width and aspect ratio
            calculated_height = int(width * h_ratio / w_ratio)
            # Use calculated height instead of input height
            height = calculated_height

        # Apply offsets to get crop position
        x_start = max(0, x_offset)
        y_start = max(0, y_offset)

        # Clamp crop bounds to image dimensions
        x_start, y_start, width, height = self._clamp_crop_bounds(
            x_start, y_start, width, height, img_width, img_height
        )

        # Perform the crop using PyTorch slicing
        # Image format: (B, H, W, C)
        cropped = image[:, y_start:y_start+height, x_start:x_start+width, :]

        return (cropped,)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "CropRX": CropRX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropRX": "CropRX",
}