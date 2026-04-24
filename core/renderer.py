"""
Core Boolean model film grain renderer.

Implements the pixel-wise algorithm from:
  Newson, Faraj, Delon, Galerne. "Realistic Film Grain Rendering."
  Image Processing On Line, 7 (2017), pp. 165-183.

The key idea: film grain IS the image, not an overlay. Silver halide crystals
are modeled as disks in a Poisson process (Boolean model). Their density is
set per-pixel to match the input gray level. The raw binary model is filtered
by a Gaussian kernel (modeling the enlarger optic + human vision) via Monte
Carlo integration, producing continuous-tone output with authentic grain texture.

The pixel-wise algorithm evaluates each output pixel independently by
regenerating grains on-the-fly from cell coordinates (no storage needed).
This is embarrassingly parallel and maps well to Numba's prange.
"""

import numpy as np
from numba import njit, prange
import math
import time

from .rng import cell_seed, rng_uniform, rng_poisson, rng_lognormal


# ---------------------------------------------------------------------------
# Internal helpers (all Numba-compiled)
# ---------------------------------------------------------------------------

@njit
def _precompute_lambda_lut(mu_r, sigma_r, n_levels=256):
    """
    Precompute Poisson intensity lambda for each possible gray level.

    lambda(u) = (1 / E[pi * r^2]) * log(1 / (1 - u_tilde))

    where u_tilde = gray_level / (n_levels + epsilon), mapped to [0, 1).
    """
    avg_area = math.pi * (mu_r * mu_r + sigma_r * sigma_r)
    lut = np.zeros(n_levels, dtype=np.float64)
    for g in range(n_levels):
        u_tilde = g / (n_levels + 0.1)
        if u_tilde < 1e-10:
            lut[g] = 0.0
        else:
            lut[g] = (1.0 / avg_area) * math.log(1.0 / (1.0 - u_tilde))
    return lut


@njit
def _precompute_mc_offsets(n_mc, filter_sigma, seed):
    """
    Pre-draw N Monte Carlo Gaussian offset vectors.

    These are shared across all output pixels (same random shifts applied
    everywhere). Each offset xi_k ~ N(0, sigma^2 * I_2), where sigma is in
    output pixel units.
    """
    offsets = np.zeros((n_mc, 2), dtype=np.float64)
    state = seed
    for k in range(n_mc):
        # Box-Muller transform for 2D Gaussian
        u1, state = rng_uniform(state)
        u2, state = rng_uniform(state)
        u1 = max(u1, 1e-30)
        r = filter_sigma * math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        offsets[k, 0] = r * math.cos(theta)
        offsets[k, 1] = r * math.sin(theta)
    return offsets


# ---------------------------------------------------------------------------
# Main rendering kernel
# ---------------------------------------------------------------------------

@njit(parallel=True)
def _render_channel_kernel(input_ch, mu_r, sigma_r, filter_sigma,
                           n_mc, zoom, global_seed):
    """
    Pixel-wise Boolean model rendering for a single channel.

    For each output pixel y:
      v(y) = (1/N) * sum_{k=1..N} 1_Z( (y + xi_k) / s )

    where 1_Z is the Boolean model indicator, xi_k are Gaussian offsets,
    and s is the zoom factor.

    Parameters
    ----------
    input_ch : uint8 array (H, W)
        Input image channel, 0-255.
    mu_r : float
        Mean grain radius in input pixel units. Typical: 0.02 - 0.15.
    sigma_r : float
        Std dev of grain radius. 0 = constant radius.
    filter_sigma : float
        Gaussian filter sigma in output pixel units. Default 0.8.
    n_mc : int
        Monte Carlo samples. More = smoother but slower. 50-200 typical.
    zoom : float
        Output zoom factor. 1.0 = same resolution as input.
    global_seed : int
        Seed for reproducibility.

    Returns
    -------
    output : float64 array (H*zoom, W*zoom) in [0, 1].
    """
    m, n = input_ch.shape
    out_m = int(m * zoom)
    out_n = int(n * zoom)
    output = np.zeros((out_m, out_n), dtype=np.float64)

    # Cell size: subdivide input pixels so each cell is ~ mu_r wide
    cells_per_pixel = max(1, int(math.ceil(1.0 / mu_r)))
    delta = 1.0 / cells_per_pixel

    # Log-normal parameters for variable grain radii
    use_variable = sigma_r > 1e-10
    ln_mu = 0.0
    ln_sigma = 0.0
    r_max = mu_r
    if use_variable:
        ratio_sq = (sigma_r / mu_r) ** 2
        ln_sigma2 = math.log(1.0 + ratio_sq)
        ln_mu = math.log(mu_r) - 0.5 * ln_sigma2
        ln_sigma = math.sqrt(ln_sigma2)
        # 99.9th percentile as maximum radius
        r_max = math.exp(ln_mu + 3.09 * ln_sigma)
    
    # How many cells away we need to check for overlapping grains
    r_max_cells = int(math.ceil(r_max / delta))

    # Precompute lambda for each gray level
    lambda_lut = _precompute_lambda_lut(mu_r, sigma_r)

    # Precompute MC offsets (shared across all pixels)
    mc_offsets = _precompute_mc_offsets(n_mc, filter_sigma, global_seed ^ 0xDEADBEEF)

    delta_sq = delta * delta

    # Main loop: parallel over output rows
    for oy in prange(out_m):
        for ox in range(out_n):
            count = 0

            for k in range(n_mc):
                # Map output pixel + MC offset to input coordinates
                x_in = (oy + mc_offsets[k, 0]) / zoom
                y_in = (ox + mc_offsets[k, 1]) / zoom

                # Determine cell neighborhood to check
                base_cx = int(math.floor(x_in / delta))
                base_cy = int(math.floor(y_in / delta))
                cx_lo = base_cx - r_max_cells
                cx_hi = base_cx + r_max_cells
                cy_lo = base_cy - r_max_cells
                cy_hi = base_cy + r_max_cells

                covered = False

                for ci in range(cx_lo, cx_hi + 1):
                    if covered:
                        break
                    for cj in range(cy_lo, cy_hi + 1):
                        if covered:
                            break

                        # Which input pixel owns this cell?
                        pi = int(math.floor(ci * delta))
                        pj = int(math.floor(cj * delta))

                        # Bounds check
                        if pi < 0 or pi >= m or pj < 0 or pj >= n:
                            continue

                        gray = input_ch[pi, pj]
                        if gray == 0:
                            continue

                        # Expected grains in this cell
                        lam_cell = lambda_lut[gray] * delta_sq
                        if lam_cell < 1e-12:
                            continue

                        # Reproducibly generate grains for cell (ci, cj)
                        seed = cell_seed(ci, cj, global_seed)
                        Q, seed = rng_poisson(lam_cell, seed)

                        for q in range(Q):
                            # Grain center: uniform within cell
                            gx_frac, seed = rng_uniform(seed)
                            gy_frac, seed = rng_uniform(seed)
                            gx = (ci + gx_frac) * delta
                            gy = (cj + gy_frac) * delta

                            # Grain radius
                            if use_variable:
                                r, seed = rng_lognormal(ln_mu, ln_sigma, seed)
                                r = min(r, r_max)
                            else:
                                r = mu_r

                            # Point-in-disk test
                            dx = x_in - gx
                            dy = y_in - gy
                            if dx * dx + dy * dy <= r * r:
                                covered = True
                                break

                if covered:
                    count += 1

            output[oy, ox] = float(count) / float(n_mc)

    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_grayscale(image, mu_r, sigma_r, filter_sigma=0.8,
                     n_mc=100, zoom=1.0, seed=42):
    """
    Render a grayscale image with film grain.

    Parameters
    ----------
    image : ndarray, uint8, shape (H, W)
        Input grayscale image.
    mu_r : float
        Mean grain radius in input pixels.
    sigma_r : float
        Std dev of grain radius (0 for constant radii).
    filter_sigma : float
        Gaussian filter sigma in output pixels.
    n_mc : int
        Monte Carlo samples per pixel.
    zoom : float
        Output zoom factor.
    seed : int
        Random seed.

    Returns
    -------
    output : uint8 array (H*zoom, W*zoom)
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    print(f"  Rendering grayscale: {image.shape[1]}x{image.shape[0]}, "
          f"mu_r={mu_r:.4f}, sigma_r={sigma_r:.4f}, "
          f"MC={n_mc}, zoom={zoom:.1f}")
    t0 = time.time()

    result = _render_channel_kernel(
        image, mu_r, sigma_r, filter_sigma, n_mc, zoom, seed
    )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def render_color(image, channel_mu_r, channel_sigma_r,
                 filter_sigma=0.8, n_mc=100, zoom=1.0, seed=42):
    """
    Render an RGB image with independent per-channel film grain.

    Real color film has three emulsion layers with different grain characteristics:
      - Red-sensitive (bottom): typically finest grains
      - Green-sensitive (middle): medium grains
      - Blue-sensitive (top): typically largest grains

    Parameters
    ----------
    image : ndarray, uint8, shape (H, W, 3)
        Input RGB image.
    channel_mu_r : list of 3 floats
        Mean grain radius per channel [R, G, B].
    channel_sigma_r : list of 3 floats
        Grain radius std dev per channel [R, G, B].
    filter_sigma : float
        Gaussian filter sigma.
    n_mc : int
        Monte Carlo samples per pixel.
    zoom : float
        Output zoom factor.
    seed : int
        Random seed (each channel uses a different derived seed).

    Returns
    -------
    output : uint8 array (H*zoom, W*zoom, 3)
    """
    assert image.ndim == 3 and image.shape[2] == 3
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    h, w = image.shape[:2]
    out_h = int(h * zoom)
    out_w = int(w * zoom)
    output = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    channel_names = ['Red', 'Green', 'Blue']

    for ch in range(3):
        print(f"  Channel {channel_names[ch]}:")
        ch_seed = seed ^ (ch * 0x1337CAFE + 7)
        result = _render_channel_kernel(
            image[:, :, ch],
            channel_mu_r[ch],
            channel_sigma_r[ch],
            filter_sigma,
            n_mc, zoom, ch_seed
        )
        output[:, :, ch] = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    return output


def warmup_jit():
    """
    Run a tiny render to trigger Numba JIT compilation.
    Call this once before timing actual renders.
    """
    tiny = np.full((8, 8), 128, dtype=np.uint8)
    _render_channel_kernel(tiny, 0.1, 0.0, 0.8, 4, 1.0, 0)