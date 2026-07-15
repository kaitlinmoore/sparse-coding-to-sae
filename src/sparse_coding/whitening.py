"""Frequency-domain whitening filter.

OWNED (whiteboard-defensible). Implements R(f) = f * exp(-(f / f0) ** 4): a
whitening ramp that flattens the ~1/f amplitude spectrum of natural images,
multiplied by a low-pass envelope that suppresses the high-frequency noise the
ramp would otherwise amplify.

Why whiten x low-pass, not whiten alone
---------------------------------------
Natural-image amplitude spectra fall off as ~1/f. To flatten (whiten) that
spectrum you multiply amplitude by f -- the bare ramp R(f) = f. But the ramp
never stops rising, so at high frequencies -- where the real image signal has
already decayed into sensor and quantization noise -- a bare ramp amplifies that
noise without bound, and the dictionary learned downstream would fit grain
instead of structure. The exp(-(f/f0)^4) factor is a low-pass that rolls the
filter off near Nyquist so noise is not blown up. The 4th power (rather than a
Gaussian's 2nd) makes the passband nearly flat and the cutoff sharp. Net effect:
a flat spectrum through the low-to-mid band where structure lives, and
suppression at the very top. Note R(0) = 0, so the filter also removes DC.

f0 is a locked hyperparameter (CLAUDE.md). Default f0 = 0.4 cycles/pixel, which
is Olshausen's sparsenet value (0.4*N cycles/image, expressed in normalized
frequency so it is independent of image size). Document it, show one sensitivity
sweep, do not tune it to prettify the filters.
"""

import numpy as np

# Locked whitening cutoff, in cycles/pixel (Nyquist = 0.5). sparsenet convention.
F0_DEFAULT = 0.4


def radial_frequency(
    shape: tuple[int, int], *, sample_spacing: float = 1.0
) -> np.ndarray:
    """Radial spatial frequency (cycles/pixel) on an unshifted ``fft2`` grid.

    Aligned with ``np.fft.fft2`` output (not ``fftshift``ed), so it multiplies
    an FFT directly. ``sample_spacing`` is the pixel pitch; the default of 1
    gives frequencies in cycles/pixel with Nyquist at 0.5.
    """
    h, w = shape
    fy = np.fft.fftfreq(h, d=sample_spacing)
    fx = np.fft.fftfreq(w, d=sample_spacing)
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    return np.sqrt(fx_grid**2 + fy_grid**2)


def whitening_filter(
    shape: tuple[int, int],
    *,
    f0: float = F0_DEFAULT,
    sample_spacing: float = 1.0,
) -> np.ndarray:
    """R(f) = f * exp(-(f/f0)^4) on the FFT grid for ``shape``.

    Real-valued 2D multiplier aligned to ``np.fft.fft2`` output. R(0) = 0.
    """
    rho = radial_frequency(shape, sample_spacing=sample_spacing)
    return rho * np.exp(-((rho / f0) ** 4))


def whiten_image(
    image: np.ndarray, *, f0: float = F0_DEFAULT, sample_spacing: float = 1.0
) -> np.ndarray:
    """Apply the whitening filter in the frequency domain.

    FFT -> multiply by R(f) -> inverse FFT -> real part. The imaginary part is
    numerical noise (a real image times a real, symmetric filter is real).
    """
    filt = whitening_filter(image.shape, f0=f0, sample_spacing=sample_spacing)
    return np.real(np.fft.ifft2(np.fft.fft2(image) * filt))


def radially_averaged_power_spectrum(
    image: np.ndarray,
    *,
    sample_spacing: float = 1.0,
    n_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """1D radial average of the 2D power spectrum ``|FFT|^2``.

    Returns ``(freqs, power)`` over radial bins from 0 to Nyquist, skipping
    empty bins. Used by both the raw-vs-whitened plots and the whitening test.
    """
    h, w = image.shape
    rho = radial_frequency(image.shape, sample_spacing=sample_spacing).ravel()
    power = (np.abs(np.fft.fft2(image)) ** 2).ravel()

    if n_bins is None:
        n_bins = min(h, w) // 2
    nyquist = 0.5 / sample_spacing
    edges = np.linspace(0.0, nyquist, n_bins + 1)
    which = np.digitize(rho, edges) - 1

    valid = (which >= 0) & (which < n_bins)
    counts = np.bincount(which[valid], minlength=n_bins)
    sums = np.bincount(which[valid], weights=power[valid], minlength=n_bins)

    keep = counts > 0
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[keep], sums[keep] / counts[keep]
