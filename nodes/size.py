"""
SizeRX: Image dimension calculator for ComfyUI.

Calculate image dimensions from aspect ratio, orientation, and megapixels.
Outputs width and height integers rounded to nearest multiple of 8.
"""

import math


class SizeRX:
    """
    SizeRX - Calculate image dimensions from aspect ratio, orientation, and megapixels.

    Dimensions are rounded to nearest multiple of 8 (required by diffusion model VAEs).

    Orientation rules:
      - landscape: width := larger dimension, height := smaller dimension
      - portrait:  height := larger dimension, width := smaller dimension

    Inputs:
        megapixels: Target megapixels (0.1 to 3.0)
        orientation: 'landscape' or 'portrait'
        preset: Aspect ratio preset (e.g., '16:9', '4:3', '1:1')

    Outputs:
        width: Calculated width in pixels
        height: Calculated height in pixels
    """

    # Major photo, cinema, and TV aspect ratios plus 2:1
    ASPECT_KEYS = [
        "1:1",      # Square
        "4:3",      # Classic TV, 35mm film frame
        "3:2",      # 35mm photography, DSLR default
        "16:9",     # HD video, modern TV/monitor
        "2:1",      # Ultrawide photography, Instagram
        "21:9",     # CinemaScope, ultrawide monitors
    ]

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = "PhotoRX/Utils"
    OUTPUT_NODE = False
    DESCRIPTION = "Calculate image dimensions from megapixels and aspect ratio. Outputs width and height rounded to nearest multiple of 8 for VAE compatibility."
    SEARCH_ALIASES = ["size", "dimensions", "resolution", "aspect ratio"]
    OUTPUT_TOOLTIPS = ("Calculated width in pixels (multiple of 8)", "Calculated height in pixels (multiple of 8)")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Megapixels: 0.1 to 3.0, step 0.1, default 1.0
                "megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "number"
                }),
                # Orientation selector
                "orientation": (["landscape", "portrait"], {"default": "landscape"}),
                # Aspect ratio preset
                "preset": (cls.ASPECT_KEYS, {"default": "16:9"}),
            }
        }

    def get_size(self, megapixels: float, orientation: str, preset: str):
        """
        Calculate width and height from megapixels and aspect ratio.

        Formula derivation:
          target_pixels = megapixels * 1,000,000
          width * height = target_pixels
          width / height = w_ratio / h_ratio

          Therefore:
            height = sqrt(target_pixels * h_ratio / w_ratio)
            width = height * (w_ratio / h_ratio)
        """
        # Parse aspect ratio (e.g., "16:9" -> 16/9) with validation
        try:
            if ":" not in preset:
                raise ValueError(f"Invalid preset format '{preset}': expected 'W:H' format")
            parts = preset.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid preset format '{preset}': expected exactly one colon")
            w_ratio, h_ratio = map(int, parts)
            if w_ratio <= 0 or h_ratio <= 0:
                raise ValueError(f"Invalid preset '{preset}': ratios must be positive integers")
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid preset format '{preset}': expected numeric values like '16:9'") from None
            raise

        # Target pixel count
        target_pixels = megapixels * 1_000_000

        # Calculate base dimensions from aspect ratio and target pixels
        base_height = math.sqrt(target_pixels * h_ratio / w_ratio)
        base_width = base_height * (w_ratio / h_ratio)

        # Round to nearest multiple of 8 (required by diffusion model VAEs)
        width = round(base_width / 8) * 8
        height = round(base_height / 8) * 8

        # Ensure minimum dimensions
        width = max(width, 8)
        height = max(height, 8)

        # Apply orientation
        if orientation == "landscape":
            # Width is the larger dimension
            final_width, final_height = max(width, height), min(width, height)
        else:
            # Portrait: height is the larger dimension
            final_width, final_height = min(width, height), max(width, height)

        return (int(final_width), int(final_height))


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SizeRX": SizeRX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SizeRX": "SizeRX",
}