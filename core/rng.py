"""
Reproducible pseudo-random number generation for cell-based grain synthesis.

The pixel-wise Boolean model algorithm requires reproducible grain generation
per spatial cell. Each cell (ci, cj) is hashed to a seed, then a local LCG
generates grains deterministically. This allows any cell's grains to be
regenerated on demand without storing them.
"""

from numba import njit
import math


@njit(inline='always')
def cell_seed(ci, cj, global_seed):
    """
    Hash cell coordinates into a reproducible RNG seed.

    Uses multiplicative hashing with large primes followed by a finalizer
    to ensure good avalanche properties (small coordinate changes produce
    completely different seeds).
    """
    h = ci * 73856093 ^ cj * 19349663 ^ global_seed
    # Murmur-style finalizer
    h = ((h >> 16) ^ h) * 0x45d9f3b
    h = ((h >> 16) ^ h) * 0x45d9f3b
    h = (h >> 16) ^ h
    return h & 0x7FFFFFFF


@njit(inline='always')
def rng_next(state):
    """Linear congruential generator step (Numerical Recipes constants)."""
    return (state * 1103515245 + 12345) & 0x7FFFFFFF


@njit(inline='always')
def rng_uniform(state):
    """Return (value in [0, 1), new_state)."""
    state = rng_next(state)
    return state / 2147483647.0, state


@njit(inline='always')
def rng_poisson(lam, state):
    """
    Sample from Poisson(lam).

    Uses the direct method (Knuth) for lam < 30, and normal approximation
    for larger values. For grain rendering, lam is typically < 1 per cell,
    so the direct method dominates.
    """
    if lam < 1e-10:
        return 0, state
    if lam < 30.0:
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            u, state = rng_uniform(state)
            p *= u
            if p <= L:
                return k, state
            k += 1
    else:
        # Normal approximation for large lambda
        u1, state = rng_uniform(state)
        u2, state = rng_uniform(state)
        u1 = max(u1, 1e-30)
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        k = max(0, int(round(lam + math.sqrt(lam) * z)))
        return k, state


@njit(inline='always')
def rng_lognormal(ln_mu, ln_sigma, state):
    """
    Sample from LogNormal distribution with log-space parameters.

    Returns (sample, new_state). The sample = exp(ln_mu + ln_sigma * Z)
    where Z ~ N(0, 1) via Box-Muller.

    ln_mu, ln_sigma are the mean and std dev of the underlying normal
    distribution (NOT the mean/std of the log-normal itself).
    """
    u1, state = rng_uniform(state)
    u2, state = rng_uniform(state)
    u1 = max(u1, 1e-30)
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return math.exp(ln_mu + ln_sigma * z), state