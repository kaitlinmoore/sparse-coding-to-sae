"""Stage 1 gate: the whitening filter flattens a synthetic 1/f power spectrum.

Synthetic because the answer is known in advance. We build a field whose
amplitude spectrum is 1/f (so its *power* spectrum falls as 1/f^2, i.e. a
log-log slope of -2), whiten it, and check the whitened power spectrum is flat
(slope ~ 0) over the passband. That is the defining property of whitening, not
an implementation detail.
"""

import numpy as np

from sparse_coding.whitening import (
    radial_frequency,
    radially_averaged_power_spectrum,
    whiten_image,
    whitening_filter,
)

F0 = 0.4


def make_one_over_f_field(n: int, *, seed: int) -> np.ndarray:
    """White noise shaped to a 1/f amplitude spectrum -> power ~ 1/f^2."""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fft2(rng.standard_normal((n, n)))
    rho = radial_frequency((n, n))
    amp = np.zeros_like(rho)
    np.divide(1.0, rho, out=amp, where=rho > 0)  # 1/f amplitude; DC stays 0
    return np.real(np.fft.ifft2(spectrum * amp))


def loglog_slope(freqs: np.ndarray, power: np.ndarray, f_lo: float, f_hi: float) -> float:
    """Slope of log10(power) vs log10(freq) over the passband [f_lo, f_hi]."""
    band = (freqs >= f_lo) & (freqs <= f_hi) & (power > 0)
    slope, _ = np.polyfit(np.log10(freqs[band]), np.log10(power[band]), 1)
    return slope


def test_synthetic_input_has_slope_minus_two():
    # Sanity: the synthetic field really is 1/f (power ~ 1/f^2) before whitening.
    field = make_one_over_f_field(256, seed=0)
    freqs, power = radially_averaged_power_spectrum(field)
    slope = loglog_slope(freqs, power, f_lo=0.03, f_hi=0.25)
    assert slope == np.float64(slope)  # not nan
    assert abs(slope - (-2.0)) < 0.3


def test_whitening_flattens_one_over_f_spectrum():
    # The actual gate: after whitening, the passband slope is ~ 0.
    field = make_one_over_f_field(256, seed=0)
    whitened = whiten_image(field, f0=F0)
    freqs, power = radially_averaged_power_spectrum(whitened)
    # Passband stays safely below f0 = 0.4, above the noisy lowest bins.
    slope = loglog_slope(freqs, power, f_lo=0.03, f_hi=0.25)
    assert abs(slope) < 0.25


def test_filter_removes_dc():
    # R(0) = 0, so the DC term is zeroed and the whitened image is ~zero-mean.
    assert whitening_filter((64, 64), f0=F0)[0, 0] == 0.0
    field = make_one_over_f_field(64, seed=1)
    assert abs(whiten_image(field, f0=F0).mean()) < 1e-9


def test_filter_matches_formula():
    # Spot-check R(f) = f * exp(-(f/f0)^4) against a direct evaluation.
    shape = (32, 48)
    rho = radial_frequency(shape)
    expected = rho * np.exp(-((rho / F0) ** 4))
    np.testing.assert_allclose(whitening_filter(shape, f0=F0), expected)
