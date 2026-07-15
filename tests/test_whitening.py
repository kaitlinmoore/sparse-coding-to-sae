"""Stage 1 gate: the whitening filter flattens a synthetic 1/f power spectrum.

Synthetic because the answer is known in advance -- generate noise with an
imposed 1/f amplitude spectrum, whiten it, and the radially-averaged spectrum
should be flat up to the low-pass cutoff f0 and rolling off above it.
"""

import pytest


@pytest.mark.skip(reason="Stage 1 not implemented")
def test_whitening_flattens_one_over_f_spectrum():
    raise NotImplementedError
