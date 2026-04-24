"""
Post-processing: perceptual grain visibility and characteristic curves.

The Boolean model produces physically correct grain density, but two additional
factors shape the look of real film that are worth modeling:

1. CHARACTERISTIC CURVE (H&D curve):
   Real film has a non-linear response to light. The S-shaped curve compresses
   shadows (toe) and highlights (shoulder), with a roughly linear middle.
   This affects where grain is most visible.

2. PERCEPTUAL VISIBILITY:
   Grain visibility depends on local contrast and tonal zone. The Boolean
   model's variance naturally follows p(1-p) (maximum at midtones), but
   real film has an asymmetry: grain is more perceptually prominent in the
   shadow-to-midtone range. Deep shadows are clean (unexposed film base),
   and dense highlights merge into smooth continuous tone.

These are applied as optional post-processing after the Boolean model render.
"""

import numpy as np


def apply_characteristic_curve(image_float, toe_strength=0.15, shoulder_strength=0.1):
    """
    Apply a simplified film characteristic curve (H&D curve).

    Maps linear input through an S-curve that compresses shadows (toe) and
    highlights (shoulder). This is applied to the INPUT image before grain
    rendering to shape how the grain density distributes across tones.

    Parameters
    ----------
    image_float : ndarray, float, [0, 1]
        Normalized input image.
    toe_strength : float
        Shadow compression. Higher = more compressed shadows.
    shoulder_strength : float
        Highlight compression. Higher = more compressed highlights.

    Returns
    -------
    curved : ndarray, float, [0, 1]
    """
    # Simple rational curve:
    # f(x) = x * (1 + toe) / (1 + toe * x) for toe
    # Then apply shoulder compression symmetrically from the top
    x = image_float.copy()

    # Toe (shadow compression)
    if toe_strength > 0:
        x = x * (1.0 + toe_strength) / (1.0 + toe_strength * x)

    # Shoulder (highlight compression)
    if shoulder_strength > 0:
        x = 1.0 - (1.0 - x) * (1.0 + shoulder_strength) / (1.0 + shoulder_strength * (1.0 - x))

    return np.clip(x, 0, 1)


def apply_visibility_modulation(rendered, original_normalized,
                                shadow_floor=0.2,
                                shadow_ramp_end=0.15,
                                highlight_ramp_start=0.5,
                                highlight_floor=0.4):
    """
    Modulate grain amplitude based on tonal zone.

    The Boolean model gives physically correct grain density, but perceptual
    grain visibility in real film prints has a specific pattern:
      - Deep shadows: clean (few exposed crystals on negative)
      - Lower midtones: grain emerges
      - Midtones: maximum grain visibility
      - Upper highlights: grain very dense but merges to smooth tone

    This function scales the grain component (deviation from expected value)
    by a zone-dependent weight, preserving the average tone while adjusting
    texture visibility.

    Parameters
    ----------
    rendered : ndarray, float, [0, 1]
        Boolean model output.
    original_normalized : ndarray, float, [0, 1]
        The expected (noise-free) tone at each pixel.
    shadow_floor : float
        Minimum grain weight in deep shadows (0 = no grain, 1 = full).
    shadow_ramp_end : float
        Luminance at which grain reaches full strength (coming up from black).
    highlight_ramp_start : float
        Luminance at which grain starts rolling off toward highlights.
    highlight_floor : float
        Minimum grain weight in bright highlights.

    Returns
    -------
    modulated : ndarray, float, [0, 1]
    """
    grain = rendered - original_normalized
    u = original_normalized

    weight = np.ones_like(u)

    # Shadow region: ramp from shadow_floor to 1.0
    shadow_mask = u < shadow_ramp_end
    if shadow_ramp_end > 0:
        t = u[shadow_mask] / shadow_ramp_end
        weight[shadow_mask] = shadow_floor + (1.0 - shadow_floor) * t

    # Highlight region: ramp from 1.0 down to highlight_floor
    highlight_mask = u > highlight_ramp_start
    if highlight_ramp_start < 1.0:
        span = 1.0 - highlight_ramp_start
        t = (u[highlight_mask] - highlight_ramp_start) / span
        t = np.clip(t, 0, 1)
        weight[highlight_mask] = 1.0 - (1.0 - highlight_floor) * t

    result = original_normalized + weight * grain
    return np.clip(result, 0, 1)


def apply_chromatic_aberration(grain_rgb, strength=0.3):
    """
    Add subtle per-channel spatial offset to simulate chromatic grain scatter.

    Real color film has slight misregistration between emulsion layers because
    they are physically separated. This produces subtle color fringing at
    grain boundaries that is part of the "film look."

    Parameters
    ----------
    grain_rgb : ndarray, float, (H, W, 3)
        Rendered color grain image.
    strength : float
        Offset in pixels. 0.2-0.5 is subtle, >1.0 is obvious.

    Returns
    -------
    shifted : ndarray, float, (H, W, 3)
    """
    if strength < 0.1:
        return grain_rgb

    result = grain_rgb.copy()
    h, w = grain_rgb.shape[:2]

    # Red: shift slightly in one direction
    shift_r = max(1, int(round(strength)))
    result[shift_r:, :, 0] = grain_rgb[:-shift_r, :, 0]

    # Blue: shift in the opposite direction
    shift_b = max(1, int(round(strength * 0.7)))
    result[:, shift_b:, 2] = grain_rgb[:, :-shift_b, 2]

    # Green stays put (it's the reference layer)
    return result