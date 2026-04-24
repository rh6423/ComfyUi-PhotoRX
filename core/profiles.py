"""
Film stock parameter profiles.

Grain parameters are calibrated to approximate the visual character of each
stock at its native ISO when scanned from 35mm. The mu_r and sigma_r values
represent the mean and standard deviation of grain radius in units of input
pixels -- meaning the apparent grain size scales with image resolution.

For a 4000 DPI scan of 35mm (roughly 5700 x 3800 pixels), these values
produce grain at approximately the right perceptual scale. For lower
resolution images (e.g., 2048 wide), you may want to increase mu_r
proportionally, or simply accept that the grain will appear finer.

sigma_r controls grain size variation (clumping). Real silver halide crystals
follow a log-normal size distribution. Higher sigma_r/mu_r ratios produce
more organic, clumpy grain (classic look). Lower ratios produce more uniform,
modern-looking grain (T-grain films).

For color stocks, the three emulsion layers have different characteristics:
  - Blue-sensitive (top layer): largest grains, most visible
  - Green-sensitive (middle): medium grains
  - Red-sensitive (bottom): finest grains, least visible
Channel parameters are given in [R, G, B] order to match numpy's RGB layout.

Parameter Provenance
--------------------
These parameters were derived through a combination of:

1. **Published measurements**: Base values informed by the Newson et al. IPOL
   2017 paper and related literature on film grain characterization.

2. **Visual calibration**: Parameters tuned by comparing rendered output to
   reference film scans (35mm, 4000 DPI) until visual characteristics matched:
   - Grain coarseness at native ISO
   - Clumping behavior (sigma_r/mu_r ratio)
   - Per-channel grain differences for color stocks

3. **Relative scaling**: Within each manufacturer's lineup, parameters scaled
   proportionally to ISO and known grain characteristics (e.g., T-grain films
   have lower sigma_r than traditional cubic crystals at the same speed).

These are approximations designed to capture the *character* of each film stock,
not exact physical measurements. For production work requiring precise film
matching, validate against your own reference scans and adjust parameters as needed.
"""


class FilmProfile:
    """A set of grain parameters characterizing a film stock."""

    def __init__(self, name, mu_r, sigma_r, filter_sigma=0.8,
                 color=False, channel_mu_r=None, channel_sigma_r=None,
                 description=""):
        self.name = name
        self.mu_r = mu_r
        self.sigma_r = sigma_r
        self.filter_sigma = filter_sigma
        self.color = color
        self.channel_mu_r = channel_mu_r or [mu_r, mu_r, mu_r]
        self.channel_sigma_r = channel_sigma_r or [sigma_r, sigma_r, sigma_r]
        self.description = description

    def __repr__(self):
        kind = "color" if self.color else "B&W"
        return (f"FilmProfile('{self.name}', {kind}, "
                f"mu_r={self.mu_r}, sigma_r={self.sigma_r})")


# ---- Black & White Stocks ----

PROFILES = {

    # --- Kodak ---

    'tri-x': FilmProfile(
        'Kodak Tri-X 400',
        mu_r=0.07, sigma_r=0.025,
        description=(
            'The iconic photojournalism film. Pronounced, irregular grain '
            'with strong clumping. Pushed to 800/1600, the grain becomes '
            'a defining visual element.'
        )
    ),

    'tmax100': FilmProfile(
        'Kodak T-Max 100',
        mu_r=0.03, sigma_r=0.005,
        description=(
            'Tabular grain (T-grain) technology. Extremely fine, uniform '
            'grain with minimal clumping. Almost digital smoothness.'
        )
    ),

    'tmax400': FilmProfile(
        'Kodak T-Max 400',
        mu_r=0.05, sigma_r=0.01,
        description=(
            'T-grain at ISO 400. Finer than Tri-X at the same speed, '
            'but with a more "modern" and less organic grain structure.'
        )
    ),

    'plus-x': FilmProfile(
        'Kodak Plus-X 125',
        mu_r=0.04, sigma_r=0.012,
        description=(
            'Classic medium-speed film with fine, traditional grain. '
            'Discontinued but beloved for portraits and landscapes.'
        )
    ),

    # --- Ilford ---

    'hp5': FilmProfile(
        'Ilford HP5 Plus 400',
        mu_r=0.065, sigma_r=0.02,
        description=(
            'Versatile ISO 400 with classic British grain character. '
            'Slightly finer than Tri-X with good clumping behavior.'
        )
    ),

    'delta100': FilmProfile(
        'Ilford Delta 100',
        mu_r=0.025, sigma_r=0.004,
        description=(
            'Core-shell crystal technology. Extremely fine grain, '
            'rivaling T-Max 100. Very smooth tonal transitions.'
        )
    ),

    'delta400': FilmProfile(
        'Ilford Delta 400',
        mu_r=0.045, sigma_r=0.009,
        description='Modern medium grain, finer than HP5 at the same speed.'
    ),

    'delta3200': FilmProfile(
        'Ilford Delta 3200',
        mu_r=0.12, sigma_r=0.05, filter_sigma=1.0,
        description=(
            'Very fast film with dramatically large grain. The grain '
            'itself becomes a textural element. Heavy clumping.'
        )
    ),

    'fp4': FilmProfile(
        'Ilford FP4 Plus 125',
        mu_r=0.035, sigma_r=0.008,
        description='Classic fine-grain ISO 125. Clean and detailed.'
    ),

    'panf': FilmProfile(
        'Ilford Pan F Plus 50',
        mu_r=0.02, sigma_r=0.003,
        description='Ultra-fine grain, almost grainless at normal enlargements.'
    ),

    # --- Fuji ---

    'acros': FilmProfile(
        'Fujifilm Neopan Acros 100',
        mu_r=0.028, sigma_r=0.005,
        description='Exceptionally fine grain with superb tonality.'
    ),

    # ---- Color Negative Stocks ----

    'portra160': FilmProfile(
        'Kodak Portra 160',
        mu_r=0.03, sigma_r=0.005,
        color=True,
        channel_mu_r=[0.025, 0.03, 0.038],
        channel_sigma_r=[0.004, 0.005, 0.008],
        description='Ultra-fine color negative. The portrait standard.'
    ),

    'portra400': FilmProfile(
        'Kodak Portra 400',
        mu_r=0.04, sigma_r=0.008,
        color=True,
        channel_mu_r=[0.035, 0.04, 0.05],
        channel_sigma_r=[0.006, 0.008, 0.012],
        description='Fine grain for ISO 400. Warm, natural colors.'
    ),

    'portra800': FilmProfile(
        'Kodak Portra 800',
        mu_r=0.06, sigma_r=0.015,
        color=True,
        channel_mu_r=[0.05, 0.06, 0.075],
        channel_sigma_r=[0.010, 0.015, 0.022],
        description='Noticeable but pleasing grain. Great in low light.'
    ),

    'ektar100': FilmProfile(
        'Kodak Ektar 100',
        mu_r=0.025, sigma_r=0.004,
        color=True,
        channel_mu_r=[0.02, 0.025, 0.032],
        channel_sigma_r=[0.003, 0.004, 0.006],
        description='World\'s finest grain color negative film.'
    ),

    'superia400': FilmProfile(
        'Fuji Superia 400',
        mu_r=0.055, sigma_r=0.015,
        color=True,
        channel_mu_r=[0.045, 0.055, 0.068],
        channel_sigma_r=[0.010, 0.015, 0.020],
        description='Consumer color film. Visible, punchy grain.'
    ),

    'gold200': FilmProfile(
        'Kodak Gold 200',
        mu_r=0.045, sigma_r=0.012,
        color=True,
        channel_mu_r=[0.038, 0.045, 0.058],
        channel_sigma_r=[0.008, 0.012, 0.016],
        description='Classic consumer film with warm cast and visible grain.'
    ),

    'ultramax400': FilmProfile(
        'Kodak Ultramax 400',
        mu_r=0.055, sigma_r=0.016,
        color=True,
        channel_mu_r=[0.045, 0.055, 0.07],
        channel_sigma_r=[0.010, 0.016, 0.022],
        description='Budget ISO 400 with characterful, visible grain.'
    ),
}


def get_profile(name):
    """Look up a film profile by key (case-insensitive)."""
    key = name.lower().replace(' ', '').replace('-', '')
    # Try exact match first
    if name.lower() in PROFILES:
        return PROFILES[name.lower()]
    # Try normalized match
    for k, v in PROFILES.items():
        if k.replace('-', '') == key:
            return v
    raise KeyError(
        f"Unknown film profile: '{name}'. "
        f"Available: {', '.join(sorted(PROFILES.keys()))}"
    )


def list_profiles():
    """Print all available film profiles."""
    print(f"\n{'Key':<16} {'Name':<30} {'Type':<6} {'mu_r':>6} {'sigma_r':>8}")
    print("-" * 72)
    for key in sorted(PROFILES.keys()):
        p = PROFILES[key]
        kind = "color" if p.color else "B&W"
        print(f"{key:<16} {p.name:<30} {kind:<6} {p.mu_r:>6.3f} {p.sigma_r:>8.4f}")
        if p.description:
            # Wrap description
            desc = p.description
            while len(desc) > 56:
                split = desc[:56].rfind(' ')
                if split < 0:
                    split = 56
                print(f"{'':>16} {desc[:split]}")
                desc = desc[split:].lstrip()
            if desc:
                print(f"{'':>16} {desc}")
    print()