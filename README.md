# PhotoRX - Film Grain Rendering for ComfyUI

A physically-based film grain renderer using the pixel-wise Boolean model. Supports both Monte Carlo simulation and fast analytical approximation.

## Features

- **Physically-based rendering**: Uses the pixel-wise Boolean model, a mathematically rigorous framework for simulating silver halide crystal distributions
- **Multiple film stocks**: Presets for popular B&W and color films (Kodak Tri-X, Ilford HP5, Portra 400, etc.)
- **Two rendering modes**: 
  - Fast analytical renderer (~100x faster, excellent visual quality)
  - Monte Carlo simulation (reference implementation, slower but physically exact)
- **Batch processing**: Works with ComfyUI's batch system for video/frame sequences
- **Per-channel grain**: Color films render separate grain layers for R/G/B channels

## Installation

### Via Manager (Recommended)

1. Open ComfyUI Manager → "Install Custom Nodes"
2. Search for "PhotoRX" or "ComfyUI-PhotoRX"
3. Click Install

### Manual Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/rh6423/ComfyUi-PhotoRX.git
pip install -r ComfyUi-PhotoRX/requirements.txt
```

## Usage

### Basic Node

Simple interface with preset selection:

```
[image] → [PhotoRX Film Grain Basic] → [output]
```

Parameters:
- **profile**: Film stock preset (default: portra400)
- **strength**: Grain intensity multiplier [0.0, 2.0] (default: 1.0)
- **black_white**: Convert to B&W before grain (default: False)
- **seed**: Random seed for reproducibility (default: 42)

### Advanced Node

Full control over all parameters:

```
[image] → [PhotoRX Film Grain Advanced] → [output]
```

Additional parameters:
- **use_fast**: Use fast analytical renderer (default: True)
- **filter_sigma**: Gaussian filter sigma [0.1, 3.0] (default: 0.8)
- **zoom**: Output zoom factor [1.0, 4.0] (default: 1.0)

## Film Stock Presets

### Black & White Films

| Key | Name | mu_r | sigma_r | Description |
|-----|------|------|---------|-------------|
| `tri-x` | Kodak Tri-X 400 | 0.070 | 0.025 | Iconic photojournalism film with pronounced, irregular grain |
| `tmax100` | Kodak T-Max 100 | 0.030 | 0.005 | Tabular grain technology, extremely fine and uniform |
| `tmax400` | Kodak T-Max 400 | 0.050 | 0.010 | Modern T-grain at ISO 400 |
| `plus-x` | Kodak Plus-X 125 | 0.040 | 0.012 | Classic medium-speed film with fine traditional grain |
| `hp5` | Ilford HP5 Plus 400 | 0.065 | 0.020 | Versatile ISO 400 with classic British grain character |
| `delta100` | Ilford Delta 100 | 0.025 | 0.004 | Core-shell crystals, extremely fine grain |
| `delta400` | Ilford Delta 400 | 0.045 | 0.009 | Modern medium grain, finer than HP5 |
| `delta3200` | Ilford Delta 3200 | 0.120 | 0.050 | Very fast film with dramatically large grain |
| `fp4` | Ilford FP4 Plus 125 | 0.035 | 0.008 | Classic fine-grain ISO 125 |
| `panf` | Ilford Pan F Plus 50 | 0.020 | 0.003 | Ultra-fine grain, almost grainless |
| `acros` | Fujifilm Neopan Acros 100 | 0.028 | 0.005 | Exceptionally fine grain with superb tonality |

### Color Negative Films

| Key | Name | Description |
|-----|------|-------------|
| `portra160` | Kodak Portra 160 | Ultra-fine color negative, the portrait standard |
| `portra400` | Kodak Portra 400 | Fine grain for ISO 400, warm natural colors |
| `portra800` | Kodak Portra 800 | Noticeable but pleasing grain, great in low light |
| `ektar100` | Kodak Ektar 100 | World's finest grain color negative film |
| `superia400` | Fuji Superia 400 | Consumer color film with visible punchy grain |
| `gold200` | Kodak Gold 200 | Classic consumer film with warm cast |
| `ultramax400` | Kodak Ultramax 400 | Budget ISO 400 with characterful visible grain |

## Technical Details

### The Pixel-Wise Boolean Model

The renderer implements the pixel-wise Boolean model as described in:
- Zhang, Wang, Tian, Pappas, "Perceptual Film Grain Synthesis", SIGGRAPH 2023
- Newson et al., "Film Grain Characterization and Synthesis", IPOL 2017

In this model, grain is represented as a random field of overlapping disks (silver halide crystals). The coverage probability at each pixel follows a Bernoulli distribution with parameter p determined by local exposure.

### Fast Analytical Renderer

The fast renderer derives the mean, variance, and spatial correlation analytically, then synthesizes grain as signal-dependent filtered Gaussian noise matching those statistics. This achieves ~100x speedup over Monte Carlo while maintaining excellent visual quality.

## Limitations

- **8-bit conversion**: The core renderer operates on uint8 data. High-bit-depth inputs are quantized to 8-bit before processing. For best results, apply grain after any tone mapping or color grading.
- **CPU-bound**: Rendering happens on CPU due to numba JIT compilation. GPU acceleration not currently supported.
- **First-run delay**: Numba JIT compilation causes a ~1-2 second delay on first render.

## Requirements

- Python 3.8+
- PyTorch (provided by ComfyUI)
- numpy
- numba (for Monte Carlo renderer)
- Pillow

## License

MIT License - see LICENSE file for details.

## Contributing

Issues and pull requests welcome! Please read the [contributing guidelines](https://github.com/rh6423/ComfyUi-PhotoRX/blob/main/.github/CONTRIBUTING.md) before submitting.

## References

1. Zhang, Y., Wang, O., Tian, J., Pappas, T. "Perceptual Film Grain Synthesis." SIGGRAPH 2023.
2. Newson, M., et al. "Film Grain Characterization and Synthesis." IPOL 2017.
3. Stoyan, D., Kendall, W., Mecke, J. "Stochastic Geometry and Its Applications." Academic Press, 1995.

---

**Project Links**: [GitHub](https://github.com/rh6423/ComfyUi-PhotoRX) | [Issues](https://github.com/rh6423/ComfyUi-PhotoRX/issues)