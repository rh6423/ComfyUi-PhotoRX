# PhotoRX - Film Grain Custom Nodes for ComfyUI

A physics-based film grain synthesis plugin that brings authentic film grain to your ComfyUI workflows. Based on the inhomogeneous Boolean model, it accurately emulates the statistical properties of real photographic film.

## Features

- **Physics-Based Rendering**: Uses the proven Boolean model from computational photography research
- **Two Renderers**: Fast analytical renderer (~100-500x faster) or Monte Carlo for reference quality
- **Film Profiles**: Pre-configured profiles for popular film stocks
- **Batch Processing**: Works with ComfyUI's batch system
- **Reproducible Results**: Seed-based randomization for consistent outputs

## Installation

### Manual Installation

1. Clone or download this repository to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/yourusername/ComfyUi-PhotoRX.git
   ```

2. Install dependencies:
   ```bash
   pip install -r ComfyUi-PhotoRX/requirements.txt
   ```

3. Restart ComfyUI

### Using ComfyUI Manager

If available, search for "PhotoRX" in the Custom Nodes Manager and install.

## Usage

After installation, restart ComfyUI. The nodes will appear under the **PhotoRX** category:

### Film Grain (Basic)

Simple interface for quick grain application.

**Inputs:**
- `image` - Input image tensor
- `film_profile` - Select a film stock (default: Tri-X 400)
- `black_white` - Convert to grayscale before applying grain
- `seed` - Random seed for reproducibility
- `zoom` - Output resolution multiplier (optional, default: 1.0)

**Outputs:**
- `image` - Image with film grain applied

### Film Grain (Advanced)

Full control over all grain parameters.

**Inputs:**
- `image` - Input image tensor
- `film_profile` - Base film stock for defaults
- `renderer` - "fast" or "monte_carlo"
- `black_white` - Convert to grayscale before applying grain
- `seed` - Random seed for reproducibility

**Optional Parameters:**
- `mu_r` - Mean grain radius (default: 0.07)
- `sigma_r` - Grain radius standard deviation (default: 0.025)
- `filter_sigma` - Gaussian filter sigma in output pixels (default: 0.8)
- `mc_samples` - Monte Carlo samples per pixel (default: 100)
- `zoom` - Output resolution multiplier (default: 1.0)

**Outputs:**
- `image` - Image with film grain applied

### Profile Info (Helper)

Displays information about a selected film profile. Useful for exploring available options.

## Film Profiles

### Black & White Films
| Profile | Description |
|---------|-------------|
| `tri-x` | Kodak Tri-X 400 - Classic high-contrast B&W |
| `hp5-plus` | Ilford HP5 Plus - Versatile all-purpose film |
| `t-max-100` | Kodak T-Max 100 - Fine grain, sharp |
| `t-max-400` | Kodak T-Max 400 - Fine grain at higher speed |
| `t-max-p3200` | Kodak T-Max P3200 - Ultra high speed |
| `delta-100` | Kodak Portra 400NC - Smooth, fine grain |
| `delta-400` | Kodak Portra 400NC - Classic portrait film |
| `pan-f-plus` | Fujifilm Acros 100 - Ultra-fine grain |
| `acros` | Fujifilm Acros 100 - Modern fine grain B&W |

### Color Films
| Profile | Description |
|---------|-------------|
| `portra-160` | Kodak Portra 160 - Fine grain portrait film |
| `portra-400` | Kodak Portra 400 - Classic portrait film |
| `ektar-100` | Kodak Ektar 100 - Vibrant colors, fine grain |
| `superia-400` | Fujifilm Superia 400 - Everyday color negative |
| `gold-200` | Kodak Gold 200 - Warm consumer film |
| `ultramax-400` | Kodak Ultramax 400 - High-speed consumer |

## Example Workflow

```
Load Image → Film Grain (Basic) → Save Image
```

Or for more control:

```
Load Image → Film Grain (Advanced) → Adjust Parameters → Save Image
```

## Performance Notes

- **Fast Renderer**: Typically completes in <1 second for HD images
- **Monte Carlo Renderer**: Can take 30-60 seconds depending on sample count and image size
- First Monte Carlo run includes JIT compilation overhead

## Troubleshooting

### "Module not found" errors
Ensure all dependencies are installed:
```bash
pip install numpy numba Pillow scipy
```

### Slow first render with Monte Carlo
This is expected - Numba JIT compiles on first use. Subsequent renders will be faster.

### Grain looks too strong/weak
Adjust the `mu_r` parameter (grain size) or use the `zoom` parameter to scale the effect.

## License

MIT License - See LICENSE file for details.

## Credits

Based on the GrainRX film grain synthesis library. The Boolean model implementation follows research from computational photography literature.

---

For issues and feature requests, please visit the [GitHub repository](https://github.com/yourusername/ComfyUi-PhotoRX/issues).