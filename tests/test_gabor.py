"""Stage 4 gate: the Gabor fit round-trips known parameters.

Synthesize a 2D Gabor with known center, orientation, spatial frequency, phase,
and envelope widths (nx, ny); fit it; recover the parameters within tolerance.
If the fitter cannot recover parameters it planted itself, the tuning
distributions downstream of it mean nothing.
"""

import pytest


@pytest.mark.skip(reason="Stage 4 not implemented")
def test_gabor_fit_recovers_known_parameters():
    raise NotImplementedError
